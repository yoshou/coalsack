#pragma once

#include <algorithm>

#include "../nn_op_node.h"

namespace coalsack {

class unsqueeze_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::vector<int64_t> axes_;
  std::string output_name_;
  std::string data_name_;
  std::string axes_input_name_;
  std::string node_name_;

 public:
  unsqueeze_node() : graph_node(), output_(std::make_shared<graph_edge>(this)), output_name_() {
    set_output(output_);
  }

  void set_axes(const std::vector<int64_t>& axes) { axes_ = axes; }

  void set_input_names(const std::string& data_name, const std::string& axes_name = "") {
    data_name_ = data_name;
    axes_input_name_ = axes_name;
  }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_node_name(const std::string& name) { node_name_ = name; }

  virtual std::string get_proc_name() const override { return "unsqueeze"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    // If input is error, propagate error to maintain sync
    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping unsqueeze [{}] (Frame: {})", node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing unsqueeze [{}] (Frame: {})", node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      // Extract input tensors by ONNX input names
      if (data_name_.empty()) {
        throw std::runtime_error("unsqueeze: data input name not set");
      }

      dynamic_tensor data = get_tensor_from_result_message(result_msg, data_name_);

      log_node_input("unsqueeze", node_name_, 0, data);

      // Get axes to unsqueeze (from second input if provided, otherwise use attribute)
      std::vector<int64_t> axes = axes_;
      if (!axes_input_name_.empty()) {
        auto axes_field = result_msg->get_field(axes_input_name_);
        if (!axes_field) {
          throw std::runtime_error("unsqueeze: axes input '" + axes_input_name_ + "' not found");
        }
        auto axes_tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(axes_field);
        if (!axes_tensor_msg) {
          throw std::runtime_error("unsqueeze: axes input is not a tensor");
        }
        if (axes_tensor_msg->get_tensor().get_dtype() != dtype::int64) {
          throw std::runtime_error("unsqueeze: axes must be int64");
        }
        const int64_t* axes_data = axes_tensor_msg->get_tensor().data_ptr<int64_t>();
        int64_t axes_size = axes_tensor_msg->get_tensor().numel();
        axes.assign(axes_data, axes_data + axes_size);
      }

      const auto& in_shape = data.shape();
      int64_t new_rank = in_shape.size() + axes.size();

      // Normalize axes to positive values
      std::vector<int64_t> normalized_axes;
      for (auto axis : axes) {
        if (axis < 0) axis += new_rank;
        if (axis < 0 || axis >= new_rank) {
          throw std::runtime_error("unsqueeze: axis out of range");
        }
        normalized_axes.push_back(axis);
      }

      std::sort(normalized_axes.begin(), normalized_axes.end());

      std::vector<int64_t> out_shape;
      size_t in_idx = 0;
      size_t axes_idx = 0;
      for (int64_t i = 0; i < new_rank; ++i) {
        if (axes_idx < normalized_axes.size() && i == normalized_axes[axes_idx]) {
          out_shape.push_back(1);
          axes_idx++;
        } else {
          out_shape.push_back(in_shape[in_idx++]);
        }
      }

      dynamic_tensor output(data.get_dtype(), out_shape);
      std::memcpy(output.data(), data.data(), data.numel() * dtype_size(data.get_dtype()));

      log_node_output("unsqueeze", node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("unsqueeze [{}]: {}", node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::unsqueeze_node, coalsack::graph_node)
