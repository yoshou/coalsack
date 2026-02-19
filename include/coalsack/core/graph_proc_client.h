#pragma once

#include <cassert>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/unordered_map.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coalsack/core/graph_proc.h"
#include "coalsack/rpc/rpc_client.h"
#include "coalsack/util/utils.h"

namespace coalsack {

class graph_proc_client {
  std::vector<std::shared_ptr<subgraph>> graphs_;
  std::unordered_map<subgraph*, std::shared_ptr<rpc_client>> rpcs_;

 public:
  void deploy(asio::io_context& io_context, std::string ipaddress, uint16_t port,
              std::shared_ptr<subgraph> g) {
    g->validate();

    std::shared_ptr<rpc_client> rpc(new rpc_client(io_context));
    graphs_.push_back(g);
    rpcs_.insert(std::make_pair(g.get(), rpc));

    rpc->connect(ipaddress, port);
    invoke_deploy(*rpc, *g);
  }

  void run() {
    initialize();

    for (auto g : graphs_) {
      invoke_run_graph(g.get());
    }
  }

  void stop() {
    for (auto g : graphs_) {
      invoke_stop_graph(g.get());
    }

    finalize();
  }

  void process(const graph_node* node, const graph_message_ptr& message) {
    process(node, "default", message);
  }

  void process(const graph_node* node, const std::string& input_name,
               const graph_message_ptr& message) {
    auto g = node->get_parent();
    auto node_idx = node->get_parent()->get_node_id(node);

    invoke_process(g, node_idx, input_name, message);
  }

 private:
  void topological_sort(std::vector<graph_node*>& nodes) {
    std::unordered_set<graph_node*> visited;
    std::vector<graph_node*> result;
    for (auto node : nodes) {
      dfs_postorder(node, visited, [&result](graph_node* node) { result.push_back(node); });
    }
    std::reverse(result.begin(), result.end());
    nodes = result;
  }

  void initialize() {
    std::vector<graph_node*> nodes;
    for (auto g : graphs_) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        nodes.push_back(node.get());
      }
    }

    topological_sort(nodes);

    for (auto node : nodes) {
      auto g = node->get_parent();
      auto node_idx = node->get_parent()->get_node_id(node);
      invoke_initialize_node(g, node_idx);
    }
  }

  void finalize() {
    std::vector<graph_node*> nodes;
    for (auto g : graphs_) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        nodes.push_back(node.get());
      }
    }

    topological_sort(nodes);

    for (auto node : nodes) {
      auto g = node->get_parent();
      auto node_idx = node->get_parent()->get_node_id(node);
      invoke_finalize_node(g, node_idx);
    }
  }

  void invoke_deploy(rpc_client& rpc, subgraph& g) {
    std::vector<uint8_t> arg, res;

    {
      std::stringstream output;
      g.save_to(output);

      std::string str = output.str();
      std::copy(str.begin(), str.end(), std::back_inserter(arg));
    }

    rpc.invoke((uint32_t)GRAPH_PROC_RPC_FUNC::DEPLOY, arg, res);
  }

  void invoke_initialize_node(subgraph* g, uint32_t node_id) {
    std::vector<uint8_t> arg, res;

    {
      std::stringstream output;
      write_uint32(output, node_id);

      std::unordered_map<std::string, subscribe_request> output_req;
      assert(node_id > 0);
      auto node = g->get_node(node_id - 1);
      std::string output_name;
      graph_edge_ptr output_edge;
      for (auto output : node->get_outputs()) {
        std::tie(output_name, output_edge) = output;
        auto req = output_edge->request;
        output_req.insert(std::make_pair(output_name, req));
      }
      {
        cereal::BinaryOutputArchive oarchive(output);
        oarchive(output_req);
      }

      std::string str = output.str();
      std::copy(str.begin(), str.end(), std::back_inserter(arg));
    }

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::INITIALIZE, arg, res);
    {
      std::stringstream input(std::string((const char*)res.data(), res.size()));

      std::unordered_map<std::string, subscribe_request> input_reqs;
      {
        cereal::BinaryInputArchive iarchive(input);
        iarchive(input_reqs);
      }

      assert(node_id > 0);
      auto node = g->get_node(node_id - 1);
      std::string input_name;
      subscribe_request req;
      for (auto input_req : input_reqs) {
        std::tie(input_name, req) = input_req;
        auto input_edge = node->get_input(input_name);
        input_edge->request = req;
      }
    }
  }

  void invoke_finalize_node(subgraph* g, uint32_t node_id) {
    std::vector<uint8_t> arg, res;

    {
      std::stringstream output;
      write_uint32(output, node_id);

      std::unordered_map<std::string, subscribe_request> output_req;
      assert(node_id > 0);
      auto node = g->get_node(node_id - 1);
      std::string output_name;
      graph_edge_ptr output_edge;
      for (auto output : node->get_outputs()) {
        std::tie(output_name, output_edge) = output;
        auto req = output_edge->request;
        output_req.insert(std::make_pair(output_name, req));
      }
      {
        cereal::BinaryOutputArchive oarchive(output);
        oarchive(output_req);
      }

      std::string str = output.str();
      std::copy(str.begin(), str.end(), std::back_inserter(arg));
    }

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::FINALIZE, arg, res);
    {
      std::stringstream input(std::string((const char*)res.data(), res.size()));

      std::unordered_map<std::string, subscribe_request> input_reqs;
      {
        cereal::BinaryInputArchive iarchive(input);
        iarchive(input_reqs);
      }

      assert(node_id > 0);
      auto node = g->get_node(node_id - 1);
      std::string input_name;
      subscribe_request req;
      for (auto input_req : input_reqs) {
        std::tie(input_name, req) = input_req;
        auto input_edge = node->get_input(input_name);
        input_edge->request = req;
      }
    }
  }

  void invoke_run_graph(subgraph* g) {
    std::vector<uint8_t> arg, res;

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::RUN, arg, res);
  }

  void invoke_stop_graph(subgraph* g) {
    std::vector<uint8_t> arg, res;

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::STOP, arg, res);
  }

  void invoke_process(subgraph* g, uint32_t node_id, const std::string& input_name,
                      const graph_message_ptr& message) {
    std::vector<uint8_t> arg, res;

    {
      std::stringstream output;
      write_uint32(output, node_id);
      write_string(output, input_name);

      {
        cereal::BinaryOutputArchive oarchive(output);
        oarchive(message);
      }

      std::string str = output.str();
      std::copy(str.begin(), str.end(), std::back_inserter(arg));
    }

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::PROCESS, arg, res);
  }
};

}  // namespace coalsack
