#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class reshape_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::string output_name_;
  std::string data_name_;
  std::string shape_name_;
  std::string node_name_;
  int64_t allowzero_;

 public:
  reshape_node()
      : graph_node(), output_(std::make_shared<graph_edge>(this)), output_name_(), allowzero_(0) {
    set_output(output_);
  }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_input_names(const std::string& data_name, const std::string& shape_name) {
    data_name_ = data_name;
    shape_name_ = shape_name;
  }

  void set_node_name(const std::string& name) { node_name_ = name; }

  void set_allowzero(int64_t allowzero) { allowzero_ = allowzero; }

  virtual std::string get_proc_name() const override { return "nn_reshape"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    // If input is error, propagate error to maintain sync
    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping reshape [{}] (Frame: {})", node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing reshape [{}] (Frame: {})", node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      // Extract input tensors by ONNX input names
      if (data_name_.empty() || shape_name_.empty()) {
        throw std::runtime_error("reshape: input names not set");
      }

      dynamic_tensor data = get_tensor_from_result_message(result_msg, data_name_);
      dynamic_tensor shape = get_tensor_from_result_message(result_msg, shape_name_);

      log_node_input("reshape", node_name_, 0, data);
      log_node_input("reshape", node_name_, 1, shape);

      // Get new shape
      const int64_t* shape_data = shape.data_ptr<int64_t>();
      int64_t shape_size = shape.numel();
      std::vector<int64_t> new_shape(shape_data, shape_data + shape_size);

      // Handle -1 in shape (infer dimension)
      int64_t infer_idx = -1;
      int64_t known_size = 1;
      for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
          if (infer_idx != -1) {
            throw std::runtime_error("reshape: only one dimension can be -1");
          }
          infer_idx = i;
        } else if (new_shape[i] == 0) {
          if (allowzero_ == 0) {
            // 0 means copy dimension from input
            new_shape[i] = data.dim(i);
            known_size *= new_shape[i];
          }
          // else: allowzero=1, treat 0 as actual dimension 0
        } else {
          known_size *= new_shape[i];
        }
      }

      if (infer_idx != -1) {
        new_shape[infer_idx] = data.numel() / known_size;
      }

      // Verify total size matches
      int64_t new_size = 1;
      for (auto dim : new_shape) new_size *= dim;
      if (new_size != data.numel()) {
        throw std::runtime_error("reshape: total size mismatch");
      }

      dynamic_tensor output(data.get_dtype(), new_shape);
      std::memcpy(output.data(), data.data(), data.numel() * dtype_size(data.get_dtype()));

      log_node_output("reshape", node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("reshape [{}]: {}", node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::reshape_node, coalsack::graph_node)
