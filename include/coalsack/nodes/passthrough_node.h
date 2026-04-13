/// @file passthrough_node.h
/// @brief Zero-copy relay node that forwards any incoming message unchanged.
/// @ingroup utility_nodes
#pragma once

#include <memory>
#include <string>

#include "coalsack/core/graph_node.h"

namespace coalsack {

/// @brief Relays every incoming message to the @b "default" output without modification.
/// @details Useful for inserting monitoring points or breaking long chains.
///
/// @par Inputs
/// - @b "default" — any @c graph_message
///
/// @par Outputs
/// - @b "default" — the same @c graph_message received on input
///
/// @par Properties
///   (none — no configurable properties)
/// @see fifo_node
class passthrough_node : public graph_node {
  graph_edge_ptr output;

 public:
  passthrough_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "passthrough"; }

  template <typename Archive>
  void serialize(Archive& archive) {}

  virtual void process([[maybe_unused]] std::string input_name,
                       graph_message_ptr message) override {
    output->send(message);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::passthrough_node, coalsack::graph_node)
