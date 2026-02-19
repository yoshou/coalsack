#pragma once

#include <spdlog/spdlog.h>

#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "coalsack/core/graph_edge.h"
#include "coalsack/core/graph_node.h"
#include "coalsack/nn/result_message.h"
#include "coalsack/tensor/dynamic_tensor.h"
#include "coalsack/tensor/dynamic_tensor_message.h"

namespace coalsack {

inline void log_node_output(const std::string& op_type, const std::string& node_name,
                            const dynamic_tensor& t) {}

inline void log_node_input(const std::string& op_type, const std::string& node_name, int index,
                           const dynamic_tensor& t) {}

inline dynamic_tensor get_tensor_from_object(const std::shared_ptr<object_message>& obj,
                                             const std::string& field_name) {
  auto field = obj->get_field(field_name);

  auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(field);
  if (tensor_msg) {
    return tensor_msg->get_tensor();
  }

  throw std::runtime_error("Field '" + field_name + "' is not a dynamic_tensor_message");
}

inline dynamic_tensor get_tensor_from_result_message(const std::shared_ptr<result_message>& msg,
                                                     const std::string& field_name) {
  auto field = msg->get_field(field_name);

  auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(field);
  if (tensor_msg) {
    return tensor_msg->get_tensor();
  }

  std::string type_name = "nullptr";
  if (field) {
    type_name = typeid(*field).name();
  }
  throw std::runtime_error("Field '" + field_name + "' is not a dynamic_tensor_message, but " +
                           type_name);
}

class unary_op_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::string op_type_;
  std::string input_name_;
  std::string output_name_;
  std::string node_name_;

  virtual dynamic_tensor compute(const dynamic_tensor& input) = 0;

 public:
  explicit unary_op_node(const std::string& op_type)
      : graph_node(),
        output_(std::make_shared<graph_edge>(this)),
        op_type_(op_type),
        input_name_(""),
        output_name_() {
    set_output(output_);
  }

  void set_input_name(const std::string& name) { input_name_ = name; }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_node_name(const std::string& name) { node_name_ = name; }

  std::string get_input_name() const { return input_name_; }
  std::string get_output_name() const { return output_name_; }

  virtual std::string get_proc_name() const override { return "nn_" + op_type_; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping {} [{}] (Frame: {})", op_type_, node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing {} [{}] (Frame: {})", op_type_, node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      dynamic_tensor input;
      if (result_msg) {
        if (input_name_.empty()) {
          throw std::runtime_error(op_type_ + ": input_name_ not set");
        }
        input = get_tensor_from_result_message(result_msg, input_name_);
      }
      log_node_input(op_type_, node_name_, 0, input);

      dynamic_tensor output = compute(input);
      log_node_output(op_type_, node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("{} [{}]: {}", op_type_, node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }
};

class binary_op_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::string node_name_;
  std::string op_type_;
  std::string input_a_name_;
  std::string input_b_name_;
  std::string output_name_;

  virtual dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) = 0;

 public:
  explicit binary_op_node(const std::string& op_type)
      : graph_node(),
        output_(std::make_shared<graph_edge>(this)),
        op_type_(op_type),
        input_a_name_(""),
        input_b_name_(""),
        output_name_() {
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "nn_" + op_type_; }

  void set_input_names(const std::string& a, const std::string& b) {
    input_a_name_ = a;
    input_b_name_ = b;
  }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_node_name(const std::string& name) { node_name_ = name; }

  std::vector<std::string> get_input_names() const { return {input_a_name_, input_b_name_}; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping {} [{}] (Frame: {})", op_type_, node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }
    spdlog::trace("Executing {} [{}] (Frame: {})", op_type_, node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      dynamic_tensor input_a, input_b;
      if (result_msg) {
        if (input_a_name_.empty() || input_b_name_.empty()) {
          throw std::runtime_error(op_type_ + ": input names not set");
        }
        input_a = get_tensor_from_result_message(result_msg, input_a_name_);
        input_b = get_tensor_from_result_message(result_msg, input_b_name_);
      }
      log_node_input(op_type_, node_name_, 0, input_a);
      log_node_input(op_type_, node_name_, 1, input_b);

      dynamic_tensor output = compute(input_a, input_b);
      log_node_output(op_type_, node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("{} [{}]: {}", op_type_, node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }
};

class variadic_op_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::string op_type_;
  std::vector<std::string> input_names_;
  std::string output_name_;
  std::string node_name_;

  virtual dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) = 0;

 public:
  variadic_op_node(const std::string& op_type, const std::vector<std::string>& input_names)
      : graph_node(),
        output_(std::make_shared<graph_edge>(this)),
        op_type_(op_type),
        input_names_(input_names),
        output_name_() {
    set_output(output_);
  }

  variadic_op_node(const std::string& op_type, size_t num_inputs)
      : graph_node(),
        output_(std::make_shared<graph_edge>(this)),
        op_type_(op_type),
        output_name_() {
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "nn_" + op_type_; }

  void set_input_names(const std::vector<std::string>& names) { input_names_ = names; }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_node_name(const std::string& name) { node_name_ = name; }

  std::vector<std::string> get_input_names() const { return input_names_; }
  std::string get_output_name() const { return output_name_; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping {} [{}] (Frame: {})", op_type_, node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }
    spdlog::trace("Executing {} [{}] (Frame: {})", op_type_, node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      std::vector<dynamic_tensor> inputs;
      if (result_msg) {
        if (input_names_.empty()) {
          throw std::runtime_error(op_type_ + ": input_names_ not set");
        }
        for (const auto& name : input_names_) {
          if (name.empty()) {
            throw std::runtime_error(op_type_ + ": input_name is empty");
          }
          inputs.push_back(get_tensor_from_result_message(result_msg, name));
        }
      }

      for (size_t i = 0; i < inputs.size(); ++i) {
        log_node_input(op_type_, node_name_, (int)i, inputs[i]);
      }

      dynamic_tensor output = compute(inputs);
      log_node_output(op_type_, node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> output_fields;
      output_fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(output_fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("{} [{}]: {}", op_type_, node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> output_fields;
      output_fields[output_name_] = nullptr;
      output_msg = result_message::error(output_fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }
};

}  // namespace coalsack
