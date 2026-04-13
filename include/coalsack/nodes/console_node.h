/// @file console_node.h
/// @brief Node that prints text_message payloads to the console.
/// @ingroup utility_nodes
#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/text_message.h"

namespace coalsack {

/// @brief Writes the text payload of incoming @c text_message objects to an output stream.
/// @details Defaults to @c std::cout.  Non-text_message input is silently discarded.
///
/// @par Inputs
/// - @b "default" — @c text_message (other subtypes are ignored)
///
/// @par Outputs
///   (none)
///
/// @par Properties
///   (none — no configurable properties)
/// @see text_message
class console_node : public graph_node {
  std::ostream* output;

 public:
  console_node() : graph_node(), output(&std::cout) {}

  virtual std::string get_proc_name() const override { return "console"; }

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}

  virtual void run() override {}

  virtual void stop() override {}

  virtual void process([[maybe_unused]] std::string input_name,
                       graph_message_ptr message) override {
    if (auto text = std::dynamic_pointer_cast<text_message>(message)) {
      (*output) << text->get_text();
    }
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::console_node, coalsack::graph_node)
