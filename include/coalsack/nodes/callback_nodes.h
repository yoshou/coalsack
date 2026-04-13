/// @file callback_nodes.h
/// @brief Nodes that bridge the graph framework with C++ callback functions.
/// @ingroup utility_nodes
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/text_message.h"
#include "coalsack/util/utils.h"

namespace coalsack {

/// @brief Named resource_base holding a list of C++ callbacks invoked on message delivery.
/// @details Register callbacks via @c add(); they are all invoked in order by @c invoke().
class annonymous_callback_list : public resource_base {
  using callback_func = std::function<void(graph_message_ptr)>;
  std::vector<callback_func> callbacks;

 public:
  virtual std::string get_name() const { return "annonymous_callback_list"; }

  void add(callback_func callback) { callbacks.push_back(callback); }

  void invoke(graph_message_ptr message) const {
    for (auto& callback : callbacks) {
      callback(message);
    }
  }
};

/// @brief Node that delivers incoming messages to registered C++ callbacks via a CHAIN edge.
/// @details At @c initialize() it resolves an @c annonymous_callback_list resource by name
///          (encoded in the output edge's @c subscribe_request data) and invokes it on each
///          incoming message.
///
/// @par Inputs
/// - @b "default" — any @c graph_message
///
/// @par Outputs
/// - @b "default" — CHAIN edge used only for subscription metadata; no data is forwarded
///
/// @par Properties
///   (none — no configurable properties)
/// @see callback_callee_node
class callback_caller_node : public graph_node {
  graph_edge_ptr output;
  std::shared_ptr<annonymous_callback_list> callbacks;

 public:
  callback_caller_node()
      : graph_node(), output(std::make_shared<graph_edge>(this, EDGE_TYPE::CHAIN)) {
    set_output(output);
  }

  virtual void initialize() override {
    auto output_req = get_output()->request;
    auto data = output_req.get_data();
    std::stringstream ss(std::string(data.begin(), data.end()));

    if (data.size() == 0) {
      return;
    }

    auto resource_name = read_string(ss);

    if (const auto resource = resources->get(resource_name)) {
      callbacks = std::dynamic_pointer_cast<annonymous_callback_list>(resource);
    }
  }

  virtual void process([[maybe_unused]] std::string input_name,
                       graph_message_ptr message) override {
    if (callbacks) {
      callbacks->invoke(message);
    }
  }

  virtual std::string get_proc_name() const override { return "callback_caller"; }

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}
};

/// @brief Source node that exposes an output edge that C++ code can push messages into.
/// @details At @c initialize() it registers an @c annonymous_callback_list resource and
///          wires it to forward any received message to @b "default" output.
///
/// @par Inputs
/// - @b "default" — any @c graph_message (injected from C++ via the internal callback)
///
/// @par Outputs
/// - @b "default" — the same @c graph_message forwarded to downstream nodes
///
/// @par Properties
///   (none — no configurable properties)
/// @see callback_caller_node
class callback_callee_node : public graph_node {
  graph_edge_ptr output;

 public:
  callback_callee_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "callback_callee"; }

  template <typename Archive>
  void serialize([[maybe_unused]] Archive& archive) {}

  virtual void initialize() override {
    auto callback_list = std::make_shared<annonymous_callback_list>();
    callback_list->add([this](graph_message_ptr message) { this->output->send(message); });

    const auto resource_name = "callback_list_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    resources->add(callback_list, resource_name);

    std::stringstream ss;
    write_string(ss, resource_name);

    std::string str = ss.str();
    std::vector<uint8_t> data(str.begin(), str.end());

    subscribe_request req;
    auto input = get_input();
    req.set_proc_name(get_proc_name());
    req.set_msg_type(text_message::get_type());
    req.set_data(data);
    input->request = req;
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::callback_caller_node, coalsack::graph_node)
COALSACK_REGISTER_NODE(coalsack::callback_callee_node, coalsack::graph_node)
