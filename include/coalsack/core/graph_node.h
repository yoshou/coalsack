#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "coalsack/core/graph_edge.h"

namespace coalsack {

class subgraph;

class resource_base {
 public:
  virtual std::string get_name() const = 0;
  virtual ~resource_base() = default;
};

class resource_list {
  std::unordered_map<std::string, std::shared_ptr<resource_base>> resources;

 public:
  void add(const std::shared_ptr<resource_base>& resource) {
    resources.insert(std::make_pair(resource->get_name(), resource));
  }

  void add(const std::shared_ptr<resource_base>& resource, const std::string& name) {
    resources.insert(std::make_pair(name, resource));
  }

  std::shared_ptr<resource_base> get(const std::string& name) const {
    const auto found = resources.find(name);
    if (found == resources.end()) {
      return nullptr;
    }
    return found->second;
  }
};

class graph_node {
 protected:
  subgraph* g;
  std::shared_ptr<resource_list> resources;

  void set_output(graph_edge_ptr output) { set_output(output, "default"); }

  void set_output(graph_edge_ptr output, std::string name) {
    output->set_name(name);
    outputs.insert(std::make_pair(name, output));
  }

 private:
  std::unordered_map<std::string, graph_edge_ptr> outputs;
  std::unordered_map<std::string, graph_edge_ptr> inputs;

 public:
  virtual std::string get_proc_name() const = 0;

  graph_node() : g(nullptr), outputs(), inputs() {}

  virtual ~graph_node() {
    std::string input_name;
    graph_edge_ptr input_edge;
    for (auto input : inputs) {
      std::tie(input_name, input_edge) = input;

      input_edge->remove_callback();
    }
  }

  void set_resources(const std::shared_ptr<resource_list>& resources) {
    this->resources = resources;
  }
  void set_input(graph_edge_ptr input) { set_input(input, "default"); }

  void set_input(graph_edge_ptr input, std::string name) {
    inputs.insert(std::make_pair(name, input));

    input->set_callback(std::make_shared<graph_message_callback>(
        [this, name](graph_message_ptr msg) { process(name, msg); }));
  }

  const std::unordered_map<std::string, graph_edge_ptr>& get_outputs() const { return outputs; }

  graph_edge_ptr get_output() const { return get_output("default"); }

  graph_edge_ptr get_output(std::string name) const {
    if (outputs.find(name) == outputs.end()) {
      throw std::invalid_argument("Output not found: " + name);
    }
    return outputs.at(name);
  }

  graph_edge_ptr get_input() const { return get_input("default"); }
  graph_edge_ptr get_input(std::string name) const {
    if (inputs.find(name) == inputs.end()) {
      throw std::invalid_argument("name");
    }
    return inputs.at(name);
  }
  const std::unordered_map<std::string, graph_edge_ptr>& get_inputs() const { return inputs; }

  virtual void process([[maybe_unused]] std::string input_name,
                       [[maybe_unused]] graph_message_ptr message) {}

  void set_parent(subgraph* g) { this->g = g; }

  subgraph* get_parent() const { return this->g; }

  virtual void initialize() {}

  virtual void finalize() {}

  virtual void run() {}

  virtual void stop() {}
};

using graph_node_ptr = std::shared_ptr<graph_node>;

template <typename T>
static void dfs_postorder(graph_node* node, std::unordered_set<graph_node*>& visited, T callback) {
  if (node == nullptr) {
    return;
  }
  if (visited.find(node) != visited.end()) {
    return;
  }

  visited.insert(node);

  for (auto input : node->get_inputs()) {
    const auto& [input_name, input_edge] = input;
    if (input_edge->get_type() != EDGE_TYPE::DATAFLOW) {
      continue;
    }
    dfs_postorder(input_edge->get_source(), visited, callback);
  }

  callback(node);
}

}  // namespace coalsack
