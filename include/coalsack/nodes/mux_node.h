#pragma once

#include <memory>
#include <string>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/object_message.h"

namespace coalsack {

class mux_node : public graph_node {
  graph_edge_ptr output;

 public:
  mux_node() : graph_node(), output(std::make_shared<graph_edge>(this)) { set_output(output); }

  virtual std::string get_proc_name() const override { return "mux"; }

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}

  virtual void process(std::string input_name, graph_message_ptr message) override {
    auto msg = std::make_shared<object_message>();
    msg->add_field(input_name, message);
    output->send(msg);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::mux_node, coalsack::graph_node)
