#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <string>
#include <unordered_map>

#include "coalsack/core/graph_message.h"

namespace coalsack {

class object_message : public graph_message {
  std::unordered_map<std::string, graph_message_ptr> fields;

 public:
  object_message() : fields() {}

  void add_field(std::string name, graph_message_ptr value) {
    fields.insert(std::make_pair(name, value));
  }
  graph_message_ptr get_field(std::string name) const { return fields.at(name); }
  void set_field(std::string name, graph_message_ptr value) { fields[name] = value; }
  const std::unordered_map<std::string, graph_message_ptr>& get_fields() const { return fields; }

  static std::string get_type() { return "object"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(fields);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::object_message, coalsack::graph_message)
