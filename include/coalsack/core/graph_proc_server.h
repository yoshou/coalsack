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
