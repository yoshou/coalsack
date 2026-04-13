/// @file graph_message.h
/// @brief Base message class and callback infrastructure for the graph framework.
/// @ingroup core_graph
#pragma once

#include <cereal/cereal.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace coalsack {

/// @defgroup core_graph Core Graph Framework
/// @brief Graph-based data-flow and control-flow processing primitives.
/// @{

/// @brief Base class for all messages passed between graph nodes.
/// @details Derived classes hold typed payload data and override @c serialize() for
///          Cereal-based binary serialization used during network transport and subgraph
///          save/load operations.
class graph_message {
 public:
  graph_message() = default;
  virtual ~graph_message() = default;

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}
};

/// @brief Shared pointer alias for graph_message.
using graph_message_ptr = std::shared_ptr<graph_message>;

/// @brief Callable wrapper stored by graph_edge to deliver messages to subscriber nodes.
struct graph_message_callback {
  using func_type = std::function<void(graph_message_ptr)>;
  func_type func;

  graph_message_callback(func_type func) : func(func) {}

  virtual void operator()(graph_message_ptr message) { func(message); }

  virtual ~graph_message_callback() = default;
};
/// @}
}  // namespace coalsack
