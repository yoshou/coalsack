#pragma once

#include <cereal/cereal.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace coalsack {
class graph_message {
 public:
  graph_message() = default;
  virtual ~graph_message() = default;

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}
};

using graph_message_ptr = std::shared_ptr<graph_message>;

struct graph_message_callback {
  using func_type = std::function<void(graph_message_ptr)>;
  func_type func;

  graph_message_callback(func_type func) : func(func) {}

  virtual void operator()(graph_message_ptr message) { func(message); }

  virtual ~graph_message_callback() = default;
};
}  // namespace coalsack
