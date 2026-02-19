#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/object_message.h"

namespace coalsack {

class demux_node : public graph_node {
 public:
  demux_node() : graph_node() {}

  graph_edge_ptr add_output(std::string name) {
    auto outputs = get_outputs();
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      auto output = std::make_shared<graph_edge>(this);
      set_output(output, name);
      return output;
    }
    return it->second;
  }

  virtual std::string get_proc_name() const override { return "demux"; }

  template <typename Archive>
  void save(Archive& archive) const {
    std::vector<std::string> output_names;
    auto outputs = get_outputs();
    for (auto output : outputs) {
      output_names.push_back(output.first);
    }
    archive(output_names);
  }

  template <typename Archive>
  void load(Archive& archive) {
    std::vector<std::string> output_names;
    archive(output_names);
    for (auto output_name : output_names) {
      set_output(std::make_shared<graph_edge>(this), output_name);
    }
  }

  virtual void process([[maybe_unused]] std::string input_name,
                       graph_message_ptr message) override {
    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (auto field : obj_msg->get_fields()) {
        try {
          get_output(field.first)->send(field.second);
        } catch (const std::invalid_argument& e) {
          spdlog::warn(e.what());
        }
      }
    }
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::demux_node, coalsack::graph_node)
