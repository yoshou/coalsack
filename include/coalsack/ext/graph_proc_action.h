/// @file graph_proc_action.h
/// @brief Action metadata annotation node.
/// @ingroup ext_nodes
#pragma once

#include <optional>
#include <string>

#include "coalsack/core/graph_proc.h"
#include "coalsack/core/graph_proc_registry.h"

namespace coalsack {

/// @brief Pass-through annotation node that attaches action metadata (id, label, icon) to the graph edge.
/// @details The node simply forwards every message unchanged, but its presence in the graph
///          allows external tools to annotate the edge with a user action identifier.
/// @par Inputs
/// - @b "default" — any @c graph_message
/// @par Outputs
/// - @b "default" — the same @c graph_message forwarded unchanged
///
/// @par Properties
/// - action_id (std::string, default "") — identifier for the associated user action
/// - label (std::string, default "") — human-readable display name for the action
/// - icon (std::string, default "") — icon identifier or path for the action
/// @see passthrough_node
class action_node : public graph_node {
  std::string action_id_;
  std::string label_;
  std::string icon_;

  graph_edge_ptr output_;

 public:
  action_node() : graph_node(), output_(std::make_shared<graph_edge>(this)) { set_output(output_); }
  virtual ~action_node() = default;

  virtual std::string get_proc_name() const override { return "action"; }

  void set_action_id(const std::string& value) { action_id_ = value; }
  std::string get_action_id() const { return action_id_; }

  void set_label(const std::string& value) { label_ = value; }
  std::string get_label() const { return label_; }

  void set_icon(const std::string& value) { icon_ = value; }
  std::string get_icon() const { return icon_; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(action_id_);
    archive(label_);
    archive(icon_);
  }

  virtual std::optional<property_value> get_property(const std::string& key) const override {
    if (key == "action_id") return action_id_;
    if (key == "label") return label_;
    if (key == "icon") return icon_;
    return std::nullopt;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    (void)input_name;
    output_->send(message);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::action_node, coalsack::graph_node)
