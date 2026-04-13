/// @file graph_node.h
/// @brief Base class for all processing nodes and graph topology utilities.
/// @ingroup core_graph
#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "coalsack/camera/camera.h"
#include "coalsack/core/graph_edge.h"
#include "coalsack/image/image.h"

namespace coalsack {

/// @brief Variant type that holds any node property value.
using property_value = std::variant<std::string, std::int64_t, double, bool, std::shared_ptr<image>,
                                    camera_t, mat4, std::vector<vec3>>;

class subgraph;

/// @brief Abstract base for shared, named resources injected into nodes at initialization.
class resource_base {
 public:
  virtual std::string get_name() const = 0;
  virtual ~resource_base() = default;
};

/// @brief Container of named resource_base instances shared across nodes in a graph.
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

/// @brief Base class for all processing nodes in a coalsack graph.
/// @details Derived classes define their input/output edges in the constructor via
///          @c set_input() / @c set_output(), and implement @c process() to react to
///          incoming messages.  The lifecycle methods @c initialize(), @c run(),
///          @c stop(), and @c finalize() are called by graph_proc in topological order.
///
/// @par Lifecycle
/// deploy → initialize → run → (process)* → stop → finalize
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

      input_edge->clear_subscribers();
    }
  }

  void set_resources(const std::shared_ptr<resource_list>& resources) {
    this->resources = resources;
  }
  void set_input(graph_edge_ptr input) { set_input(input, "default"); }

  void set_input(graph_edge_ptr input, std::string name) {
    inputs.insert(std::make_pair(name, input));

    input->add_subscriber(std::make_shared<graph_message_callback>(
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

  /// @brief Called by the framework when a message arrives on the named input port.
  /// @param input_name Name of the input port that received the message.
  /// @param message    The received message (cast to the expected derived type as needed).
  virtual void process([[maybe_unused]] std::string input_name,
                       [[maybe_unused]] graph_message_ptr message) {}

  void set_parent(subgraph* g) { this->g = g; }

  subgraph* get_parent() const { return this->g; }

  virtual void initialize() {}

  virtual void finalize() {}

  virtual void run() {}

  virtual void stop() {}

  /// @brief Returns the value of a named property, or @c std::nullopt if not found.
  virtual std::optional<property_value> get_property(
      [[maybe_unused]] const std::string& key) const {
    return std::nullopt;
  }
};

/// @brief Shared pointer alias for graph_node.
using graph_node_ptr = std::shared_ptr<graph_node>;

/// @brief Depth-first post-order traversal of the input sub-graph rooted at @p node.
/// @tparam T Callable with signature @c void(graph_node*).
/// @param node    Starting node (may be @c nullptr).
/// @param visited Set of already-visited nodes (accumulates across calls).
/// @param callback Invoked for each node after all its DATAFLOW predecessors.
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

/// @brief Partitions @p nodes into weakly connected components of the DATAFLOW sub-graph.
/// @param nodes All nodes to consider.
/// @return A vector of components, each component being a vector of node pointers.
inline std::vector<std::vector<graph_node*>> compute_weakly_connected_components(
    const std::vector<graph_node*>& nodes) {
  std::unordered_set<graph_node*> node_set(nodes.begin(), nodes.end());
  std::unordered_map<graph_node*, std::vector<graph_node*>> adjacency;
  for (auto* node : nodes) {
    adjacency[node];
  }

  for (auto* node : nodes) {
    for (const auto& [input_name, input_edge] : node->get_inputs()) {
      (void)input_name;
      if (input_edge->get_type() != EDGE_TYPE::DATAFLOW) {
        continue;
      }

      auto* source = input_edge->get_source();
      if (source == nullptr || node_set.find(source) == node_set.end()) {
        continue;
      }

      adjacency[node].push_back(source);
      adjacency[source].push_back(node);
    }
  }

  std::vector<std::vector<graph_node*>> components;
  std::unordered_set<graph_node*> visited;
  for (auto* start : nodes) {
    if (visited.find(start) != visited.end()) {
      continue;
    }

    std::queue<graph_node*> queue;
    std::vector<graph_node*> component;
    visited.insert(start);
    queue.push(start);

    while (!queue.empty()) {
      auto* node = queue.front();
      queue.pop();
      component.push_back(node);

      for (auto* next : adjacency[node]) {
        if (visited.insert(next).second) {
          queue.push(next);
        }
      }
    }

    components.push_back(std::move(component));
  }

  return components;
}

}  // namespace coalsack
