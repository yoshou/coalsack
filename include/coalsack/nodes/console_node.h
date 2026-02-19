#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/text_message.h"

namespace coalsack {

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
