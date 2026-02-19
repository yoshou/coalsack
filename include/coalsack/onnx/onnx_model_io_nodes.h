#pragma once

#include "coalsack/core/graph_edge.h"
#include "coalsack/core/graph_node.h"
#include "coalsack/nn/result_message.h"
#include "coalsack/tensor/dynamic_tensor.h"
#include "coalsack/tensor/dynamic_tensor_message.h"

namespace coalsack {

class model_input_node : public graph_node {
 private:
  std::unordered_map<std::string, dynamic_tensor> tensors_;
  graph_edge_ptr output_;
  uint64_t frame_number_;

 public:
  model_input_node() : graph_node(), output_(std::make_shared<graph_edge>(this)), frame_number_(1) {
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "model_input"; }

  void set_tensor(const std::string& name, const dynamic_tensor& tensor) {
    tensors_[name] = tensor;
  }
  void set_frame_number(uint64_t fn) { frame_number_ = fn; }
  graph_edge_ptr get_output() const { return output_; }

  virtual void run() override {
    std::shared_ptr<result_message> result;
    try {
      std::unordered_map<std::string, graph_message_ptr> fields;

      for (const auto& [name, tensor] : tensors_) {
        auto tensor_msg = std::make_shared<dynamic_tensor_message>(tensor);
        fields[name] = tensor_msg;
      }

      result = result_message::ok(fields);
      result->set_frame_number(frame_number_);

    } catch (const std::exception& e) {
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (const auto& [name, tensor] : tensors_) {
        fields[name] = nullptr;
      }
      result = result_message::error(fields, e.what());
      result->set_frame_number(frame_number_);
    }
    output_->send(result);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {}
};

class model_output_node : public graph_node {
 private:
  std::unordered_map<std::string, dynamic_tensor> outputs_;
  std::function<void(const std::unordered_map<std::string, dynamic_tensor>&)> callback_;

 public:
  model_output_node() : graph_node(), outputs_(), callback_(nullptr) {}

  virtual std::string get_proc_name() const override { return "model_output"; }

  void set_callback(
      std::function<void(const std::unordered_map<std::string, dynamic_tensor>&)> cb) {
    callback_ = cb;
  }

  const std::unordered_map<std::string, dynamic_tensor>& get_collected_outputs() const {
    return outputs_;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    try {
      auto result_msg = std::dynamic_pointer_cast<result_message>(message);
      if (!result_msg) {
        throw std::runtime_error("Expected result_message");
      }

      if (result_msg->is_error()) {
        throw std::runtime_error("Model output received error: " + result_msg->get_error_message());
      }

      outputs_.clear();
      for (const auto& [name, field_msg] : result_msg->get_fields()) {
        auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(field_msg);
        if (tensor_msg) {
          outputs_[name] = tensor_msg->get_tensor();
        }
      }

      if (callback_) {
        callback_(outputs_);
      }

    } catch (const std::exception& e) {
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::model_input_node, coalsack::graph_node)
COALSACK_REGISTER_NODE(coalsack::model_output_node, coalsack::graph_node)
