#pragma once

#include <string>

#include "coalsack/core/graph_message.h"

namespace coalsack {

class number_message : public graph_message {
  double value;

 public:
  number_message() : value(0.0) {}

  void set_value(double value) { this->value = value; }
  double get_value() const { return value; }
  static std::string get_type() { return "number"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(value);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::number_message, coalsack::graph_message)
