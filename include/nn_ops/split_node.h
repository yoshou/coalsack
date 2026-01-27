#pragma once

#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class split_node : public unary_op_node {
 public:
  split_node() : unary_op_node("split"), axis_(0) {}

  void set_axis(int64_t axis) { axis_ = axis; }
  void set_splits(const std::vector<int64_t>& splits) { splits_ = splits; }

  void set_output_names(const std::vector<std::string>& names) { output_names_ = names; }

  std::vector<std::string> output_names_;

  void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    // If input is error, propagate error to maintain sync
    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping split [{}] (Frame: {})", node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (size_t i = 0; i < output_names_.size(); ++i) {
        fields[output_names_[i]] = nullptr;
      }
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing split [{}] (Frame: {})", node_name_, frame_number);

    std::shared_ptr<result_message> out_result;
    try {
      // Extract input tensor by ONNX input name
      if (input_name_.empty()) {
        throw std::runtime_error("split: input name not set");
      }

      dynamic_tensor input = get_tensor_from_result_message(result_msg, input_name_);
      log_node_input("split", node_name_, 0, input);

      // Split the tensor
      auto outputs = split_impl(input);

      // Send all outputs in a single result_message
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto out_tensor_msg = std::make_shared<dynamic_tensor_message>(outputs[i]);
        std::string field_name =
            (i < output_names_.size()) ? output_names_[i] : ("output_" + std::to_string(i));
        fields[field_name] = out_tensor_msg;
      }

      out_result = result_message::ok(fields);
      out_result->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("split [{}]: {}", "", e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (size_t i = 0; i < output_names_.size(); ++i) {
        fields[output_names_[i]] = nullptr;
      }
      out_result = result_message::error(fields, e.what());
      out_result->set_frame_number(frame_number);
    }
    output_->send(out_result);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    auto outputs = split_impl(input);
    return outputs.empty() ? dynamic_tensor() : outputs.front();
  }

 private:
  int64_t axis_;
  std::vector<int64_t> splits_;

  template <typename T>
  void split_data(const dynamic_tensor& input, dynamic_tensor& output, int64_t outer_size,
                  int64_t inner_size, int64_t src_axis_offset, int64_t split_size,
                  int64_t axis_dim) const {
    const T* src = input.data_ptr<T>();
    T* dst = output.data_ptr<T>();

    for (int64_t outer = 0; outer < outer_size; ++outer) {
      int64_t src_offset = outer * axis_dim * inner_size + src_axis_offset * inner_size;
      int64_t dst_offset = outer * split_size * inner_size;
      int64_t copy_size = split_size * inner_size;
      std::memcpy(dst + dst_offset, src + src_offset, copy_size * sizeof(T));
    }
  }

  std::vector<dynamic_tensor> split_impl(const dynamic_tensor& input) const {
    const auto& in_shape = input.shape();
    int64_t axis = axis_ < 0 ? in_shape.size() + axis_ : axis_;

    if (axis < 0 || axis >= static_cast<int64_t>(in_shape.size())) {
      throw std::runtime_error("split: axis out of range");
    }

    // If splits not specified, split evenly
    std::vector<int64_t> splits = splits_;
    if (splits.empty()) {
      int64_t num_outputs = static_cast<int64_t>(output_names_.size());
      if (num_outputs == 0) num_outputs = 2;
      int64_t split_size = in_shape[axis] / num_outputs;
      splits.assign(num_outputs, split_size);
    }

    // Verify splits sum to axis dimension
    int64_t sum = 0;
    for (auto s : splits) sum += s;
    if (sum != in_shape[axis]) {
      throw std::runtime_error("split: splits do not sum to axis dimension");
    }

    std::vector<dynamic_tensor> outputs;
    outputs.reserve(splits.size());

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      outer_size *= in_shape[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < static_cast<int64_t>(in_shape.size()); ++i) {
      inner_size *= in_shape[i];
    }

    // Split data
    int64_t src_axis_offset = 0;
    for (size_t i = 0; i < splits.size(); ++i) {
      std::vector<int64_t> out_shape = in_shape;
      out_shape[axis] = splits[i];
      dynamic_tensor split_output(input.get_dtype(), out_shape);

      if (input.get_dtype() == dtype::float32) {
        split_data<float>(input, split_output, outer_size, inner_size, src_axis_offset, splits[i],
                          in_shape[axis]);
      } else if (input.get_dtype() == dtype::float64) {
        split_data<double>(input, split_output, outer_size, inner_size, src_axis_offset, splits[i],
                           in_shape[axis]);
      } else if (input.get_dtype() == dtype::int64) {
        split_data<int64_t>(input, split_output, outer_size, inner_size, src_axis_offset, splits[i],
                            in_shape[axis]);
      } else if (input.get_dtype() == dtype::int32) {
        split_data<int32_t>(input, split_output, outer_size, inner_size, src_axis_offset, splits[i],
                            in_shape[axis]);
      } else {
        throw std::runtime_error("split: unsupported dtype");
      }

      src_axis_offset += splits[i];
      outputs.push_back(split_output);
    }

    return outputs;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::split_node, coalsack::graph_node)
