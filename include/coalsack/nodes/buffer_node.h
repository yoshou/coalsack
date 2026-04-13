/// @file buffer_node.h
/// @brief Buffering node that caches the latest received message and republishes it periodically.
/// @ingroup utility_nodes
#pragma once

#include <algorithm>
#include <cereal/types/base_class.hpp>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

#include "coalsack/messages/list_message.h"
#include "coalsack/messages/object_message.h"
#include "coalsack/nodes/heartbeat_node.h"

namespace coalsack {

/// @brief Stores the most recent incoming message and re-sends it on every heartbeat tick.
/// @details Thread-safe via an internal mutex.  Extends heartbeat_node so the heartbeat
///          interval controls the re-publish rate.
///
/// @par Inputs
/// - @b "default" — any @c graph_message (the message to buffer)
///
/// @par Outputs
/// - @b "default" — the most recently received @c graph_message
///
/// @par Properties
/// - interval (uint32_t, inherited from heartbeat_node) — re-publish period in milliseconds
/// @see heartbeat_node, clock_buffer_node
class buffer_node : public heartbeat_node {
  graph_edge_ptr output;
  graph_message_ptr message;
  std::mutex mtx;

 public:
  buffer_node() : heartbeat_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "buffer"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<heartbeat_node>(this));
  }

  virtual void process([[maybe_unused]] std::string input_name,
                       graph_message_ptr message) override {
    std::lock_guard<std::mutex> lock(mtx);

    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      if (!this->message) {
        this->message = message;
      } else {
        auto current_obj_msg = std::dynamic_pointer_cast<object_message>(this->message);
        for (auto& [name, field] : obj_msg->get_fields()) {
          current_obj_msg->set_field(name, field);
        }
      }
    } else if (auto list_msg = std::dynamic_pointer_cast<list_message>(message)) {
      if (!this->message) {
        this->message = message;
      } else {
        auto current_list_msg = std::dynamic_pointer_cast<list_message>(this->message);

        const auto copy_size = std::min(current_list_msg->length(), list_msg->length());
        std::size_t i = 0;
        for (; i < copy_size; i++) {
          current_list_msg->set(i, list_msg->get(i));
        }
        for (; i < list_msg->length(); i++) {
          current_list_msg->add(list_msg->get(i));
        }
      }
    } else {
      this->message = message;
    }
  }

  virtual void tick() override {
    graph_message_ptr message;
    {
      std::lock_guard<std::mutex> lock(mtx);
      message = this->message;
    }
    output->send(message);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::buffer_node, coalsack::heartbeat_node)
