#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class topk_node : public graph_node {
 public:
  topk_node()
      : graph_node(),
        output_(std::make_shared<graph_edge>(this)),
        op_type_("topk"),
        axis_(-1),
        largest_(true),
        data_name_(),
        k_name_(),
        values_output_name_(),
        indices_output_name_(),
        node_name_() {
    set_output(output_);
  }

  void set_axis(int64_t axis) { axis_ = axis; }
  void set_largest(bool largest) { largest_ = largest; }

  void set_values_output_name(const std::string& name) { values_output_name_ = name; }
  void set_indices_output_name(const std::string& name) { indices_output_name_ = name; }

  void set_input_names(const std::string& data_name, const std::string& k_name) {
    data_name_ = data_name;
    k_name_ = k_name;
  }

  void set_node_name(const std::string& name) { node_name_ = name; }

  virtual std::string get_proc_name() const override { return "topk"; }

  void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    // If input is error, propagate error to maintain sync
    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping {} [{}] (Frame: {})", op_type_, node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[values_output_name_] = nullptr;
      fields[indices_output_name_] = nullptr;
      auto result = result_message::error(fields, "Input error");
      result->set_frame_number(frame_number);
      output_->send(result);
      return;
    }

    spdlog::trace("Executing {} [{}] (Frame: {})", op_type_, node_name_, frame_number);

    std::shared_ptr<result_message> result;
    try {
      // Extract input tensors by names
      if (data_name_.empty() || k_name_.empty()) {
        throw std::runtime_error(op_type_ + ": input names not set");
      }

      dynamic_tensor input = get_tensor_from_result_message(result_msg, data_name_);
      dynamic_tensor k_tensor = get_tensor_from_result_message(result_msg, k_name_);

      log_node_input(op_type_, node_name_, 0, input);
      log_node_input(op_type_, node_name_, 1, k_tensor);

      auto [values, indices] = compute_topk(input, k_tensor);

      log_node_output(op_type_, node_name_ + " (Values)", values);
      log_node_output(op_type_, node_name_ + " (Indices)", indices);

      auto values_msg = std::make_shared<dynamic_tensor_message>(values);
      auto indices_msg = std::make_shared<dynamic_tensor_message>(indices);

      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[values_output_name_] = values_msg;
      fields[indices_output_name_] = indices_msg;

      result = result_message::ok(fields);
      result->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("{} [{}]: {}", op_type_, node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[values_output_name_] = nullptr;
      fields[indices_output_name_] = nullptr;
      result = result_message::error(fields, e.what());
      result->set_frame_number(frame_number);
    }
    output_->send(result);
  }

 private:
  graph_edge_ptr output_;
  std::string op_type_;
  int64_t axis_;
  bool largest_;
  std::string data_name_;
  std::string k_name_;
  std::string values_output_name_;
  std::string indices_output_name_;
  std::string node_name_;

  std::pair<dynamic_tensor, dynamic_tensor> compute_topk(const dynamic_tensor& input,
                                                         const dynamic_tensor& k_tensor) {
    int64_t k = *k_tensor.data_ptr<int64_t>();
    const auto& in_shape = input.shape();
    int64_t axis = axis_ < 0 ? in_shape.size() + axis_ : axis_;

    std::vector<int64_t> out_shape = in_shape;
    out_shape[axis] = k;

    dynamic_tensor values(input.get_dtype(), out_shape);
    dynamic_tensor indices(dtype::int64, out_shape);

    if (input.get_dtype() == dtype::float32) {
      topk_impl<float>(input, values, indices, k, axis);
    } else if (input.get_dtype() == dtype::float64) {
      topk_impl<double>(input, values, indices, k, axis);
    } else if (input.get_dtype() == dtype::int64) {
      topk_impl<int64_t>(input, values, indices, k, axis);
    } else if (input.get_dtype() == dtype::int32) {
      topk_impl<int32_t>(input, values, indices, k, axis);
    } else {
      throw std::runtime_error("topk: unsupported dtype");
    }

    return {values, indices};
  }

  template <typename T>
  void topk_impl(const dynamic_tensor& input, dynamic_tensor& values, dynamic_tensor& indices,
                 int64_t k, int64_t axis) {
    const T* in_data = input.data_ptr<T>();
    T* val_data = values.data_ptr<T>();
    int64_t* idx_data = indices.data_ptr<int64_t>();

    const auto& in_shape = input.shape();
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) outer_size *= in_shape[i];
    int64_t axis_size = in_shape[axis];
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < static_cast<int64_t>(in_shape.size()); ++i)
      inner_size *= in_shape[i];

    for (int64_t outer = 0; outer < outer_size; ++outer) {
      for (int64_t inner = 0; inner < inner_size; ++inner) {
        std::vector<std::pair<T, int64_t>> pairs;
        for (int64_t a = 0; a < axis_size; ++a) {
          int64_t idx = outer * axis_size * inner_size + a * inner_size + inner;
          pairs.push_back({in_data[idx], a});
        }

        if (largest_) {
          std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                            [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
          std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        for (int64_t i = 0; i < k; ++i) {
          int64_t out_idx = outer * k * inner_size + i * inner_size + inner;
          val_data[out_idx] = pairs[i].first;
          idx_data[out_idx] = pairs[i].second;
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::topk_node, coalsack::graph_node)
