/// @file clock_buffer_node.h
/// @brief Buffer node driven by an external clock signal instead of an internal timer.
/// @ingroup utility_nodes
#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "coalsack/core/graph_node.h"

namespace coalsack {

/// @brief Caches the latest incoming message and republishes it whenever a clock pulse arrives.
/// @details Unlike buffer_node, timing is controlled by an external edge (@b "clock") rather
///          than a dedicated thread, enabling synchronisation with upstream frame sources.
///
/// @par Inputs
/// - @b "default" — any @c graph_message (the payload to buffer)
/// - @b "clock"   — any @c graph_message (triggers a re-publish of the latest buffered message)
///
/// @par Outputs
/// - @b "default" — the most recently buffered @c graph_message
///
/// @par Properties
///   (none — no configurable properties)
/// @see buffer_node, heartbeat_node
class clock_buffer_node : public graph_node {
  graph_edge_ptr output;
  graph_message_ptr message;
  std::mutex mtx;

 protected:
  graph_edge_ptr get_clock() const { return get_input("clock"); }

 public:
  clock_buffer_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  void set_clock(graph_edge_ptr clock) { set_input(clock, "clock"); }

  virtual std::string get_proc_name() const override { return "clock_buffer"; }

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "default") {
      std::lock_guard<std::mutex> lock(mtx);
      this->message = message;
    } else if (input_name == "clock") {
      graph_message_ptr message;
      {
        std::lock_guard<std::mutex> lock(mtx);
        message = this->message;
      }
      output->send(message);
    }
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::clock_buffer_node, coalsack::graph_node)
