#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "coalsack/core/subgraph.h"

namespace coalsack {

enum class GRAPH_PROC_RPC_FUNC : uint32_t {
  DEPLOY = 0,
  INITIALIZE = 1,
  RUN = 2,
  STOP = 3,
  FINALIZE = 4,
  PROCESS = 5,
};

class graph_proc {
  std::shared_ptr<subgraph> g;
  std::shared_ptr<resource_list> resources;

 public:
  graph_proc(const std::shared_ptr<resource_list>& resources = std::make_shared<resource_list>())
      : g(new subgraph()), resources(resources) {}

  std::shared_ptr<subgraph> get_graph() const { return g; }

  std::shared_ptr<resource_list> get_resources() const { return resources; }

  void deploy(const std::shared_ptr<subgraph>& g) { this->g = g; }

  void run() {
    initialize();
    run(g.get());
  }

  void stop() {
    stop(g.get());
    finalize();
  }

  void process(const graph_node* node, const graph_message_ptr& message) {
    process(node, "default", message);
  }

  void process(const graph_node* node, const std::string& input_name,
               const graph_message_ptr& message) {
    auto g = node->get_parent();
    auto node_id = node->get_parent()->get_node_id(node);

    if (node_id > 0 && node_id <= g->get_node_count()) {
      auto node = g->get_node(node_id - 1);
      node->process(input_name, message);
    }
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
    for (uint32_t i = 0; i < g->get_node_count(); i++) {
      auto node = g->get_node(i);
      nodes.push_back(node.get());
    }

    topological_sort(nodes);

    for (auto node : nodes) {
      node->set_resources(resources);
      node->initialize();
    }
  }

  void finalize() {
    for (uint32_t i = 0; i < g->get_node_count(); i++) {
      auto node = g->get_node(i);
      node->finalize();
    }
  }

  void run(subgraph* g) {
    for (uint32_t i = 0; i < g->get_node_count(); i++) {
      auto node = g->get_node(i);
      node->run();
    }
  }

  void stop(subgraph* g) {
    for (uint32_t i = 0; i < g->get_node_count(); i++) {
      auto node = g->get_node(i);
      node->stop();
    }
  }
};

}  // namespace coalsack
