#pragma once

#include <cassert>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/unordered_map.hpp>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
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

  using request_map = std::unordered_map<std::string, subscribe_request>;
  using node_request_map = std::unordered_map<uint32_t, request_map>;

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

  void initialize() {
    std::vector<graph_node*> nodes;
    for (auto g : graphs_) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        nodes.push_back(node.get());
      }
    }

    execute_batches(nodes, true);
  }

  void run() {
    for (auto g : graphs_) {
      invoke_run_graph(g.get());
    }
  }

  void stop() {
    for (auto g : graphs_) {
      invoke_stop_graph(g.get());
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

    execute_batches(nodes, false);
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
  struct batch_info {
    subgraph* graph;
    std::vector<graph_node*> nodes;
  };

  void topological_sort(std::vector<graph_node*>& nodes) {
    std::unordered_set<graph_node*> visited;
    std::vector<graph_node*> result;
    for (auto node : nodes) {
      dfs_postorder(node, visited, [&result](graph_node* node) { result.push_back(node); });
    }
    std::reverse(result.begin(), result.end());
    nodes = result;
  }

  static request_map collect_output_reqs(const graph_node* node, EDGE_TYPE edge_type) {
    request_map output_reqs;
    for (const auto& [output_name, output_edge] : node->get_outputs()) {
      if (output_edge->get_type() != edge_type) {
        continue;
      }
      output_reqs.insert(std::make_pair(output_name, output_edge->request));
    }
    return output_reqs;
  }

  static void apply_input_reqs(const std::shared_ptr<graph_node>& node, const request_map& reqs,
                               EDGE_TYPE edge_type) {
    for (const auto& [input_name, req] : reqs) {
      auto input_edge = node->get_input(input_name);
      if (input_edge->get_type() != edge_type) {
        continue;
      }
      input_edge->request = req;
    }
  }

  std::vector<batch_info> build_batches(const std::vector<graph_node*>& nodes) {
    std::vector<batch_info> batches;
    const auto components = compute_weakly_connected_components(nodes);
    if (components.empty()) {
      return batches;
    }

    std::unordered_map<graph_node*, size_t> component_index;
    batches.reserve(components.size());
    for (size_t i = 0; i < components.size(); i++) {
      auto& component_nodes = components[i];
      if (component_nodes.empty()) {
        continue;
      }

      auto* parent = component_nodes.front()->get_parent();
      batches.push_back(batch_info{parent, component_nodes});
      for (auto* node : component_nodes) {
        component_index[node] = i;
      }
    }

    std::vector<std::set<size_t>> deps(components.size());
    for (auto* node : nodes) {
      const auto target_index = component_index.at(node);
      for (const auto& [input_name, input_edge] : node->get_inputs()) {
        (void)input_name;
        if (input_edge->get_type() != EDGE_TYPE::CHAIN) {
          continue;
        }

        auto* source = input_edge->get_source();
        if (source == nullptr) {
          continue;
        }

        const auto source_found = component_index.find(source);
        if (source_found == component_index.end()) {
          continue;
        }

        const auto source_index = source_found->second;
        if (source_index == target_index) {
          continue;
        }

        deps[source_index].insert(target_index);
      }
    }

    std::vector<batch_info> ordered_batches;
    ordered_batches.reserve(batches.size());
    std::vector<bool> emitted(batches.size(), false);
    std::vector<bool> visiting(batches.size(), false);

    std::function<void(size_t)> visit = [&](size_t index) {
      if (emitted[index]) {
        return;
      }
      if (visiting[index]) {
        throw std::runtime_error("Circular dependency in CHAIN edge graph");
      }

      visiting[index] = true;
      for (auto dep : deps[index]) {
        visit(dep);
      }
      visiting[index] = false;
      emitted[index] = true;
      ordered_batches.push_back(batches[index]);
    };

    for (size_t i = 0; i < batches.size(); i++) {
      visit(i);
    }

    return ordered_batches;
  }

  void execute_batches(const std::vector<graph_node*>& nodes, bool initialize_batch) {
    for (const auto& batch : build_batches(nodes)) {
      std::vector<uint32_t> node_ids;
      node_ids.reserve(batch.nodes.size());
      node_request_map output_reqs_by_node;

      for (auto* node : batch.nodes) {
        const auto node_id = batch.graph->get_node_id(node);
        node_ids.push_back(node_id);

        auto output_reqs = collect_output_reqs(node, EDGE_TYPE::CHAIN);
        if (!output_reqs.empty()) {
          output_reqs_by_node.insert(std::make_pair(node_id, std::move(output_reqs)));
        }
      }

      node_request_map input_reqs_by_node;
      if (initialize_batch) {
        input_reqs_by_node = invoke_batch_initialize(batch.graph, node_ids, output_reqs_by_node);
      } else {
        input_reqs_by_node = invoke_batch_finalize(batch.graph, node_ids, output_reqs_by_node);
      }

      for (const auto& [node_id, input_reqs] : input_reqs_by_node) {
        auto node = batch.graph->get_node(node_id - 1);
        apply_input_reqs(node, input_reqs, EDGE_TYPE::CHAIN);
      }
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

  node_request_map invoke_batch_initialize(subgraph* g, const std::vector<uint32_t>& node_ids,
                                           const node_request_map& output_reqs_by_node) {
    std::vector<uint8_t> arg, res;

    {
      std::stringstream output;
      write_uint32(output, static_cast<uint32_t>(node_ids.size()));
      for (auto node_id : node_ids) {
        write_uint32(output, node_id);
      }
      {
        cereal::BinaryOutputArchive oarchive(output);
        oarchive(output_reqs_by_node);
      }

      const auto str = output.str();
      std::copy(str.begin(), str.end(), std::back_inserter(arg));
    }

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::BATCH_INITIALIZE, arg, res);

    node_request_map input_reqs_by_node;
    std::stringstream input(std::string((const char*)res.data(), res.size()));
    cereal::BinaryInputArchive iarchive(input);
    iarchive(input_reqs_by_node);
    return input_reqs_by_node;
  }

  node_request_map invoke_batch_finalize(subgraph* g, const std::vector<uint32_t>& node_ids,
                                         const node_request_map& output_reqs_by_node) {
    std::vector<uint8_t> arg, res;

    {
      std::stringstream output;
      write_uint32(output, static_cast<uint32_t>(node_ids.size()));
      for (auto node_id : node_ids) {
        write_uint32(output, node_id);
      }
      {
        cereal::BinaryOutputArchive oarchive(output);
        oarchive(output_reqs_by_node);
      }

      const auto str = output.str();
      std::copy(str.begin(), str.end(), std::back_inserter(arg));
    }

    auto rpc = rpcs_.at(g);
    rpc->invoke((uint32_t)GRAPH_PROC_RPC_FUNC::BATCH_FINALIZE, arg, res);

    node_request_map input_reqs_by_node;
    std::stringstream input(std::string((const char*)res.data(), res.size()));
    cereal::BinaryInputArchive iarchive(input);
    iarchive(input_reqs_by_node);
    return input_reqs_by_node;
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
