/// @file mux_node.h
/// @brief Multiplexer node that wraps any incoming message in an object_message labelled by port name.
/// @ingroup utility_nodes
#pragma once

#include <memory>
#include <string>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/object_message.h"

namespace coalsack {

/// @brief Accepts messages on any named input port and re-emits each as an @c object_message
///        keyed by the originating port name.
/// @details Downstream nodes can inspect @c object_message::get_field(port_name) to identify
///          which upstream input was active.
///
/// @par Inputs
/// - @b "{any}" — any @c graph_message on arbitrarily named ports
///
/// @par Outputs
/// - @b "default" — @c object_message with a single field whose key is the input port name
///
/// @par Properties
///   (none — no configurable properties)
/// @see demux_node
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
