#pragma once

#include <string>

#include "coalsack/core/graph_message.h"

namespace coalsack {

class text_message : public graph_message {
  std::string text;

 public:
  text_message() : text() {}

  void set_text(std::string text) { this->text = text; }
  std::string get_text() const { return text; }
  static std::string get_type() { return "text"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(text);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::text_message, coalsack::graph_message)
