#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "coalsack/core/graph_message.h"

namespace coalsack {

class list_message : public graph_message {
  std::vector<graph_message_ptr> list;

 public:
  list_message() : list() {}

  void add(graph_message_ptr value) { list.push_back(value); }
  graph_message_ptr get(size_t idx) const { return list[idx]; }
  void set(size_t idx, graph_message_ptr value) { list[idx] = value; }
  std::size_t length() const { return list.size(); }
  static std::string get_type() { return "list"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(list);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::list_message, coalsack::graph_message)
