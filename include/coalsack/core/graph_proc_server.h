#pragma once

#include <spdlog/spdlog.h>

#include <cassert>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/unordered_map.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "coalsack/core/graph_proc.h"
#include "coalsack/rpc/rpc_server.h"
#include "coalsack/util/utils.h"

namespace coalsack {

class graph_proc_server {
  rpc_server rpc_server_;
  std::shared_ptr<resource_list> resources_;
  std::map<uint32_t, std::shared_ptr<subgraph>> graphs_;
  std::mutex mtx;

  using request_map = std::unordered_map<std::string, subscribe_request>;
  using node_request_map = std::unordered_map<uint32_t, request_map>;

  static request_map collect_input_reqs(const std::shared_ptr<graph_node>& node,
                                        EDGE_TYPE edge_type) {
    request_map input_reqs;
    for (const auto& [input_name, input_edge] : node->get_inputs()) {
      if (input_edge->get_type() != edge_type) {
        continue;
      }
      input_reqs.insert(std::make_pair(input_name, input_edge->request));
    }
    return input_reqs;
  }

  static void apply_output_reqs(const std::shared_ptr<graph_node>& node, const request_map& reqs,
                                EDGE_TYPE edge_type) {
    for (const auto& [output_name, req] : reqs) {
      auto output = node->get_output(output_name);
      if (output->get_type() != edge_type) {
        continue;
      }
      output->request = req;
    }
  }

  static void topological_sort(std::vector<graph_node*>& nodes) {
    std::unordered_set<graph_node*> visited;
    std::vector<graph_node*> result;
    for (auto* node : nodes) {
      dfs_postorder(node, visited, [&result](graph_node* current) { result.push_back(current); });
    }
    std::reverse(result.begin(), result.end());
    nodes = std::move(result);
  }

 public:
  graph_proc_server(
      boost::asio::io_context& io_context, std::string address, uint16_t port,
      const std::shared_ptr<resource_list>& resources = std::make_shared<resource_list>())
      : rpc_server_(io_context, address, port), resources_(resources) {
    rpc_server_.register_handler((uint32_t)GRAPH_PROC_RPC_FUNC::DEPLOY,
                                 [this](uint32_t session, const std::vector<uint8_t>& arg,
                                        [[maybe_unused]] std::vector<uint8_t>& res) -> uint32_t {
                                   spdlog::debug("Deploy graph (session = {0})", session);

                                   std::stringstream ss(std::string(arg.begin(), arg.end()));
                                   std::shared_ptr<subgraph> g_(new subgraph());
                                   g_->load_from(ss);

                                   {
                                     std::lock_guard<std::mutex> lock(mtx);
                                     graphs_.insert(std::make_pair(session, g_));
                                   }

                                   return 0;
                                 });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::INITIALIZE,
        [this](uint32_t session, const std::vector<uint8_t>& arg,
               std::vector<uint8_t>& res) -> uint32_t {
          spdlog::debug("Initialize graph (session = {0})", session);

          std::stringstream arg_ss(std::string((const char*)arg.data(), arg.size()));
          auto node_id = read_uint32(arg_ss);

          auto g_ = graphs_.at(session);

          assert(node_id > 0);
          auto node = g_->get_node(node_id - 1);

          std::unordered_map<std::string, subscribe_request> output_reqs;
          {
            cereal::BinaryInputArchive iarchive(arg_ss);
            iarchive(output_reqs);
          }

          std::string output_name;
          subscribe_request req;
          for (auto output_req : output_reqs) {
            std::tie(output_name, req) = output_req;
            auto output = node->get_output(output_name);
            output->request = req;
          }

          node->set_resources(resources_);
          node->initialize();

          std::unordered_map<std::string, subscribe_request> input_reqs;
          std::string input_name;
          graph_edge_ptr input_edge;
          for (auto input : node->get_inputs()) {
            std::tie(input_name, input_edge) = input;
            auto req = input_edge->request;
            input_reqs.insert(std::make_pair(input_name, req));
          }

          std::stringstream res_ss;
          {
            cereal::BinaryOutputArchive oarchive(res_ss);
            oarchive(input_reqs);
          }

          std::string res_s = res_ss.str();
          std::copy(res_s.begin(), res_s.end(), std::back_inserter(res));

          return 0;
        });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::BATCH_INITIALIZE,
        [this](uint32_t session, const std::vector<uint8_t>& arg,
               std::vector<uint8_t>& res) -> uint32_t {
          spdlog::debug("Batch initialize graph (session = {0})", session);

          std::stringstream arg_ss(std::string((const char*)arg.data(), arg.size()));
          const auto node_count = read_uint32(arg_ss);

          std::vector<uint32_t> node_ids;
          node_ids.reserve(node_count);
          for (uint32_t i = 0; i < node_count; i++) {
            node_ids.push_back(read_uint32(arg_ss));
          }

          node_request_map output_reqs_by_node;
          {
            cereal::BinaryInputArchive iarchive(arg_ss);
            iarchive(output_reqs_by_node);
          }

          auto g_ = graphs_.at(session);

          std::vector<graph_node*> batch_nodes;
          batch_nodes.reserve(node_ids.size());
          for (auto node_id : node_ids) {
            assert(node_id > 0);
            auto node = g_->get_node(node_id - 1);
            batch_nodes.push_back(node.get());

            const auto found = output_reqs_by_node.find(node_id);
            if (found != output_reqs_by_node.end()) {
              apply_output_reqs(node, found->second, EDGE_TYPE::CHAIN);
            }
          }

          topological_sort(batch_nodes);

          for (auto* batch_node : batch_nodes) {
            const auto node_id = g_->get_node_id(batch_node);
            auto node = g_->get_node(node_id - 1);
            node->set_resources(resources_);
            node->initialize();
          }

          node_request_map input_reqs_by_node;
          for (auto node_id : node_ids) {
            auto node = g_->get_node(node_id - 1);
            auto input_reqs = collect_input_reqs(node, EDGE_TYPE::CHAIN);
            if (!input_reqs.empty()) {
              input_reqs_by_node.insert(std::make_pair(node_id, std::move(input_reqs)));
            }
          }

          std::stringstream res_ss;
          {
            cereal::BinaryOutputArchive oarchive(res_ss);
            oarchive(input_reqs_by_node);
          }

          const auto res_s = res_ss.str();
          std::copy(res_s.begin(), res_s.end(), std::back_inserter(res));

          return 0;
        });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::RUN,
        [this](uint32_t session, [[maybe_unused]] const std::vector<uint8_t>& arg,
               [[maybe_unused]] std::vector<uint8_t>& res) -> uint32_t {
          spdlog::debug("Run graph (session = {0})", session);

          auto g_ = graphs_.at(session);
          g_->run();

          return 0;
        });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::STOP,
        [this](uint32_t session, [[maybe_unused]] const std::vector<uint8_t>& arg,
               [[maybe_unused]] std::vector<uint8_t>& res) -> uint32_t {
          spdlog::debug("Stop graph (session = {0})", session);

          auto g_ = graphs_.at(session);
          g_->stop();

          return 0;
        });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::FINALIZE,
        [this](uint32_t session, const std::vector<uint8_t>& arg,
               std::vector<uint8_t>& res) -> uint32_t {
          spdlog::debug("Finalize graph (session = {0})", session);

          std::stringstream arg_ss(std::string((const char*)arg.data(), arg.size()));
          auto node_id = read_uint32(arg_ss);

          auto g_ = graphs_.at(session);

          assert(node_id > 0);
          auto node = g_->get_node(node_id - 1);

          std::unordered_map<std::string, subscribe_request> output_reqs;
          {
            cereal::BinaryInputArchive iarchive(arg_ss);
            iarchive(output_reqs);
          }

          std::string output_name;
          subscribe_request req;
          for (auto output_req : output_reqs) {
            std::tie(output_name, req) = output_req;
            auto output = node->get_output(output_name);
            output->request = req;
          }

          node->finalize();

          std::unordered_map<std::string, subscribe_request> input_reqs;
          std::string input_name;
          graph_edge_ptr input_edge;
          for (auto input : node->get_inputs()) {
            std::tie(input_name, input_edge) = input;
            auto req = input_edge->request;
            input_reqs.insert(std::make_pair(input_name, req));
          }

          std::stringstream res_ss;
          {
            cereal::BinaryOutputArchive oarchive(res_ss);
            oarchive(input_reqs);
          }

          std::string res_s = res_ss.str();
          std::copy(res_s.begin(), res_s.end(), std::back_inserter(res));

          return 0;
        });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::BATCH_FINALIZE,
        [this](uint32_t session, const std::vector<uint8_t>& arg,
               std::vector<uint8_t>& res) -> uint32_t {
          spdlog::debug("Batch finalize graph (session = {0})", session);

          std::stringstream arg_ss(std::string((const char*)arg.data(), arg.size()));
          const auto node_count = read_uint32(arg_ss);

          std::vector<uint32_t> node_ids;
          node_ids.reserve(node_count);
          for (uint32_t i = 0; i < node_count; i++) {
            node_ids.push_back(read_uint32(arg_ss));
          }

          node_request_map output_reqs_by_node;
          {
            cereal::BinaryInputArchive iarchive(arg_ss);
            iarchive(output_reqs_by_node);
          }

          auto g_ = graphs_.at(session);

          std::vector<graph_node*> batch_nodes;
          batch_nodes.reserve(node_ids.size());
          for (auto node_id : node_ids) {
            assert(node_id > 0);
            auto node = g_->get_node(node_id - 1);
            batch_nodes.push_back(node.get());

            const auto found = output_reqs_by_node.find(node_id);
            if (found != output_reqs_by_node.end()) {
              apply_output_reqs(node, found->second, EDGE_TYPE::CHAIN);
            }
          }

          topological_sort(batch_nodes);

          for (auto* batch_node : batch_nodes) {
            const auto node_id = g_->get_node_id(batch_node);
            auto node = g_->get_node(node_id - 1);
            node->finalize();
          }

          node_request_map input_reqs_by_node;
          for (auto node_id : node_ids) {
            auto node = g_->get_node(node_id - 1);
            auto input_reqs = collect_input_reqs(node, EDGE_TYPE::CHAIN);
            if (!input_reqs.empty()) {
              input_reqs_by_node.insert(std::make_pair(node_id, std::move(input_reqs)));
            }
          }

          std::stringstream res_ss;
          {
            cereal::BinaryOutputArchive oarchive(res_ss);
            oarchive(input_reqs_by_node);
          }

          const auto res_s = res_ss.str();
          std::copy(res_s.begin(), res_s.end(), std::back_inserter(res));

          return 0;
        });

    rpc_server_.register_handler(
        (uint32_t)GRAPH_PROC_RPC_FUNC::PROCESS,
        [this](uint32_t session, const std::vector<uint8_t>& arg,
               [[maybe_unused]] std::vector<uint8_t>& res) -> uint32_t {
          std::stringstream arg_ss(std::string((const char*)arg.data(), arg.size()));
          const auto node_id = read_uint32(arg_ss);
          const auto input_name = read_string(arg_ss);
          spdlog::debug("Process node (session = {0}, node = {1}, input_name = {2})", session,
                        node_id, input_name);

          std::shared_ptr<graph_message> msg;
          {
            cereal::BinaryInputArchive iarchive(arg_ss);
            iarchive(msg);
          }

          auto g_ = graphs_.at(session);

          assert(node_id > 0);
          auto node = g_->get_node(node_id - 1);

          node->process(input_name, msg);

          return 0;
        });

    rpc_server_.on_discconect([this](uint32_t session) {
      spdlog::debug("Delete graph (session = {0})", session);

      {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = graphs_.find(session);
        if (it != graphs_.end()) {
          it->second->stop();
          graphs_.erase(it);
        }
      }
    });
  }

  uint16_t get_port() const { return rpc_server_.local_endpoint().port(); }
};

}  // namespace coalsack
