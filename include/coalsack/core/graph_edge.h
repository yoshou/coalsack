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

enum class EDGE_TYPE {
  DATAFLOW = 0,
  CHAIN = 1,
};

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

class graph_edge {
  graph_node* source;
  std::string name;
  EDGE_TYPE edge_type;
  std::vector<std::shared_ptr<graph_message_callback>> callbacks;

 public:
  graph_edge(graph_node* source, EDGE_TYPE edge_type = EDGE_TYPE::DATAFLOW)
      : source(source), name(), edge_type(edge_type), callbacks() {}

  void set_name(std::string name) { this->name = name; }

  std::string get_name() const { return name; }

  EDGE_TYPE get_type() const { return edge_type; }

  graph_node* get_source() const { return source; }

  void set_callback(std::shared_ptr<graph_message_callback> callback) {
    if (std::find(callbacks.begin(), callbacks.end(), callback) != callbacks.end()) {
      throw std::logic_error("The callback has been already registerd");
    }
    callbacks.push_back(callback);
  }

  void remove_callback() { callbacks.clear(); }

  void send(graph_message_ptr message) {
    for (const auto& callback : callbacks) {
      if (callback) {
        (*callback)(message);
      }
    }
  }

  subscribe_request request;
};

using graph_edge_ptr = std::shared_ptr<graph_edge>;

}  // namespace coalsack
