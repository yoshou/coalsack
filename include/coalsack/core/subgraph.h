#pragma once

#include <algorithm>
#include <cassert>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coalsack/core/graph_node.h"

namespace coalsack {

class subgraph {
  std::vector<graph_node_ptr> nodes;
  std::unordered_map<const graph_node*, uint32_t> node_ids;

 public:
  subgraph() {}

  void add_node(graph_node_ptr node) {
    if (node->get_parent()) {
      throw std::invalid_argument("The node already belongs to another graph.");
    }
    node_ids.insert(std::make_pair(node.get(), (uint32_t)nodes.size() + 1));
    nodes.push_back(node);
    node->set_parent(this);
  }

  uint32_t get_node_id(const graph_node* node) const { return node_ids.at(node); }

  uint32_t get_node_count() const { return nodes.size(); }

  graph_node_ptr get_node(uint32_t idx) const { return nodes[idx]; }

  template <typename Archive>
  void save(Archive& archive) const {
    archive(nodes);

    using node_port = std::tuple<uint32_t, std::string>;

    std::vector<std::tuple<node_port, node_port, EDGE_TYPE>> edges;
    for (size_t i = 0; i < nodes.size(); i++) {
      auto target = nodes[i];

      std::string target_input_name;
      graph_edge_ptr target_input;
      for (auto input : target->get_inputs()) {
        std::tie(target_input_name, target_input) = input;

        if (target_input->get_type() == EDGE_TYPE::DATAFLOW) {
          node_port target_port((uint32_t)i, target_input_name);

          auto source = target_input->get_source();
          auto source_output_name = target_input->get_name();

          auto j = std::distance(nodes.begin(), std::find_if(nodes.begin(), nodes.end(),
                                                             [source](graph_node_ptr node) {
                                                               return node.get() == source;
                                                             }));

          assert(nodes[j].get() == source);

          node_port source_port((uint32_t)j, source_output_name);

          edges.push_back(std::make_tuple(source_port, target_port, EDGE_TYPE::DATAFLOW));
        } else if (target_input->get_type() == EDGE_TYPE::CHAIN) {
          node_port target_port((uint32_t)i, target_input_name);

          auto source_output_name = target_input->get_name();

          node_port source_port((uint32_t)-1, source_output_name);

          edges.push_back(std::make_tuple(source_port, target_port, EDGE_TYPE::CHAIN));
        }
      }
    }

    archive(edges);
  }

  template <class Archive>
  void load(Archive& archive) {
    archive(nodes);

    for (size_t i = 0; i < nodes.size(); i++) {
      auto node = nodes[i];
      node_ids.insert(std::make_pair(node.get(), (uint32_t)i + 1));
      node->set_parent(this);
    }

    using node_port = std::tuple<uint32_t, std::string>;

    std::vector<std::tuple<node_port, node_port, EDGE_TYPE>> edges;
    archive(edges);

    for (auto edge : edges) {
      node_port source_port, target_port;
      EDGE_TYPE edge_type;
      std::tie(source_port, target_port, edge_type) = edge;

      if (edge_type == EDGE_TYPE::DATAFLOW) {
        auto j = std::get<0>(source_port);
        auto i = std::get<0>(target_port);

        auto source = nodes[j];
        auto target = nodes[i];

        auto target_input_name = std::get<1>(target_port);
        auto source_output_name = std::get<1>(source_port);

        target->set_input(source->get_output(source_output_name), target_input_name);
      } else if (edge_type == EDGE_TYPE::CHAIN) {
        auto j = std::get<0>(source_port);
        auto i = std::get<0>(target_port);

        assert(j == (uint32_t)-1);

        auto target = nodes[i];

        auto target_input_name = std::get<1>(target_port);
        auto source_output_name = std::get<1>(source_port);

        auto external_output = std::make_shared<graph_edge>(nullptr);
        external_output->set_name(source_output_name);

        target->set_input(external_output, target_input_name);
      }
    }
  }

  void save_to(std::ostream& s) const {
    cereal::BinaryOutputArchive oarchive(s);
    oarchive(*this);
  }

  void load_from(std::istream& s) {
    cereal::BinaryInputArchive iarchive(s);
    iarchive(*this);
  }

  void run() {
    std::unordered_set<graph_node*> visited;
    std::vector<graph_node*> ordered_nodes;
    for (auto& node : nodes) {
      dfs_postorder(node.get(), visited,
                    [&ordered_nodes](graph_node* node) { ordered_nodes.push_back(node); });
    }
    std::reverse(ordered_nodes.begin(), ordered_nodes.end());
    for (auto node : ordered_nodes) {
      node->run();
    }
  }

  void stop() {
    for (auto node : nodes) {
      node->stop();
    }
  }

  void validate() {
    for (size_t i = 0; i < nodes.size(); i++) {
      auto target = nodes[i];

      std::string target_input_name;
      graph_edge_ptr target_input;
      for (auto input : target->get_inputs()) {
        std::tie(target_input_name, target_input) = input;

        if (target_input->get_type() == EDGE_TYPE::DATAFLOW) {
          auto source = target_input->get_source();
          if (source->get_parent() != this) {
            throw std::logic_error("Invalid input");
          }
        }
      }
    }
  }

  void merge(const subgraph& other) {
    for (uint32_t i = 0; i < other.get_node_count(); i++) {
      auto node = other.get_node(i);
      node->set_parent(nullptr);
      add_node(node);
    }
  }
};

class subgraph_node : public graph_node {
  std::shared_ptr<subgraph> g;
  std::unordered_map<std::string, graph_edge_ptr> output_map;

 public:
  subgraph_node() : graph_node(), g(std::make_shared<subgraph>()) {}

  subgraph& get_subgraph() const { return *g; }

  virtual std::string get_proc_name() const override { return "subgraph"; }

  void set_output(graph_edge_ptr output, std::string name) {
    output_map[name] = output;
    graph_node::set_output(output, name);
  }

  graph_edge_ptr get_output(std::string name) const {
    if (output_map.find(name) == output_map.end()) {
      throw std::invalid_argument("name");
    }
    return output_map.at(name);
  }

  graph_edge_ptr find_output(std::string name) const {
    if (output_map.find(name) == output_map.end()) {
      return nullptr;
    }
    return output_map.at(name);
  }

  virtual void initialize() override {
    const auto outputs = get_outputs();
    if (g) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        node->set_resources(resources);
        node->initialize();
      }
    }
  }

  virtual void finalize() override {
    if (g) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        node->finalize();
      }
    }
  }

  virtual void run() override {
    if (g) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        node->run();
      }
    }
  }

  virtual void stop() override {
    if (g) {
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        node->stop();
      }
    }
  }

  template <typename Archive>
  void save(Archive& archive) const {
    archive(g);

    auto output_size = static_cast<uint32_t>(output_map.size());
    archive(output_size);
    for (const auto& [name, output] : output_map) {
      archive(name);
      uint32_t node_id = 0;
      for (uint32_t i = 0; i < g->get_node_count(); i++) {
        auto node = g->get_node(i);
        if (node.get() == output->get_source()) {
          node_id = i + 1;
          break;
        }
      }
      archive(node_id);
    }
  }

  template <class Archive>
  void load(Archive& archive) {
    archive(g);

    uint32_t output_size = 0;
    archive(output_size);
    for (uint32_t i = 0; i < output_size; i++) {
      std::string name;
      uint32_t node_id = 0;
      archive(name);
      archive(node_id);
      auto node = g->get_node(node_id - 1);
      auto output = node->get_output();
      output_map[name] = output;
      graph_node::set_output(output, name);
    }
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::subgraph_node, coalsack::graph_node)
