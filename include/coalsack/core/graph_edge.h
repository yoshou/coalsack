/// @file graph_edge.h
/// @brief Directed edge connecting two graph nodes for message delivery.
/// @ingroup core_graph
#pragma once

#include <algorithm>
#include <cereal/cereal.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "coalsack/core/graph_message.h"

namespace coalsack {

class graph_node;

/// @brief Specifies the semantics of a graph edge.
enum class EDGE_TYPE {
  DATAFLOW = 0, ///< Data-flow edge: message carrying the primary payload between nodes.
  CHAIN = 1,    ///< Chain edge: ordering-only edge without data payload (for sequencing).
};

/// @brief Serializable descriptor sent from a downstream node to request an upstream
///        subscription over an RPC channel.
class subscribe_request {
  std::string proc_name;
  std::string msg_type;
  std::vector<uint8_t> data;

 public:
  subscribe_request() {}

  void set_proc_name(std::string value) { proc_name = value; }
  std::string get_proc_name() const { return proc_name; }

  void set_msg_type(std::string value) { msg_type = value; }
  std::string get_msg_type() const { return msg_type; }

  void set_data(const std::vector<uint8_t>& value) { data = value; }
  std::vector<uint8_t> get_data() const { return data; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(proc_name, msg_type, data);
  }
};

/// @brief Directed edge in a processing graph that fans out messages to all subscribers.
/// @details A graph_edge is owned by the source node and shared with zero or more
///          downstream (target) nodes.  When the source calls @c send(), every
///          registered callback receives the message in registration order.
///          @c add_subscriber() and @c clear_subscribers() are private and accessible
///          only to graph_node via the friend declaration.
class graph_edge {
  graph_node* source;
  std::string name;
  EDGE_TYPE edge_type;
  std::vector<std::shared_ptr<graph_message_callback>> callbacks;

  void add_subscriber(std::shared_ptr<graph_message_callback> callback) {
    if (std::find(callbacks.begin(), callbacks.end(), callback) != callbacks.end()) {
      throw std::logic_error("The callback has been already registerd");
    }
    callbacks.push_back(callback);
  }

  void clear_subscribers() { callbacks.clear(); }

 public:
  friend class graph_node;

  graph_edge(graph_node* source, EDGE_TYPE edge_type = EDGE_TYPE::DATAFLOW)
      : source(source), name(), edge_type(edge_type), callbacks() {}

  void set_name(std::string name) { this->name = name; }

  std::string get_name() const { return name; }

  EDGE_TYPE get_type() const { return edge_type; }

  graph_node* get_source() const { return source; }

  /// @brief Broadcasts @p message to all registered subscriber callbacks.
  /// @param message The message to deliver.
  void send(graph_message_ptr message) {
    for (const auto& callback : callbacks) {
      if (callback) {
        (*callback)(message);
      }
    }
  }

  subscribe_request request;
};

/// @brief Shared pointer alias for graph_edge.
using graph_edge_ptr = std::shared_ptr<graph_edge>;

}  // namespace coalsack
