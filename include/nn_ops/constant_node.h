#pragma once

#include "../nn_op_node.h"
#include "dynamic_tensor.h"
#include "dynamic_tensor_message.h"
#include "graph_proc.h"
#include "result_message.h"

namespace coalsack {

class constant_node : public graph_node {
 private:
  graph_edge_ptr output_;
  dynamic_tensor value_;
  std::string value_name_;

 public:
  constant_node() : graph_node(), value_(), value_name_() {
    output_ = std::make_shared<graph_edge>(this);
    set_output(output_);
  }

  constant_node(dynamic_tensor value, const std::string& value_name)
      : graph_node(), value_(std::move(value)), value_name_(value_name) {
    output_ = std::make_shared<graph_edge>(this);
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "nn_constant"; }

  void set_value(dynamic_tensor value) { value_ = std::move(value); }

  void set_value_name(const std::string& name) { value_name_ = name; }

  const dynamic_tensor& get_value() const { return value_; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    (void)input_name;
    uint64_t frame_number = 0;
    if (auto result_msg = std::dynamic_pointer_cast<result_message>(message)) {
      frame_number = result_msg->get_frame_number();
    }

    spdlog::trace("Executing Constant:{} (Frame: {})", value_name_, frame_number);

    log_node_output("Constant", value_name_, value_);

    auto tensor_msg = std::make_shared<dynamic_tensor_message>(value_);

    std::unordered_map<std::string, graph_message_ptr> fields;
    fields[value_name_] = tensor_msg;
    auto result = result_message::ok(fields);
    result->set_frame_number(frame_number);
    output_->send(result);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::constant_node, coalsack::graph_node)
