#pragma once

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class slice_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::string output_name_;
  std::string data_name_;
  std::string starts_name_;
  std::string ends_name_;
  std::string axes_name_;
  std::string steps_name_;
  std::string node_name_;

 public:
  slice_node() : graph_node(), output_(std::make_shared<graph_edge>(this)), output_name_() {
    set_output(output_);
  }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_input_names(const std::string& data, const std::string& starts, const std::string& ends,
                       const std::string& axes = "", const std::string& steps = "") {
    data_name_ = data;
    starts_name_ = starts;
    ends_name_ = ends;
    axes_name_ = axes;
    steps_name_ = steps;
  }

  void set_node_name(const std::string& name) { node_name_ = name; }

  virtual std::string get_proc_name() const override { return "nn_slice"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    // If input is error, propagate error to maintain sync
    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping slice [{}] (Frame: {})", node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing slice [{}] (Frame: {})", node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      // Extract input tensors by ONNX input names
      if (data_name_.empty() || starts_name_.empty() || ends_name_.empty()) {
        throw std::runtime_error("slice: required input names not set");
      }

      dynamic_tensor data = get_tensor_from_result_message(result_msg, data_name_);
      dynamic_tensor starts = get_tensor_from_result_message(result_msg, starts_name_);
      dynamic_tensor ends = get_tensor_from_result_message(result_msg, ends_name_);

      log_node_input("slice", node_name_, 0, data);

      std::vector<int64_t> axes;
      std::vector<int64_t> steps;

      if (!axes_name_.empty()) {
        auto axes_field = result_msg->get_field(axes_name_);
        if (axes_field) {
          auto axes_tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(axes_field);
          if (axes_tensor_msg) {
            const int64_t* axes_data = axes_tensor_msg->get_tensor().data_ptr<int64_t>();
            int64_t axes_size = axes_tensor_msg->get_tensor().numel();
            axes.assign(axes_data, axes_data + axes_size);
          }
        }
      }

      if (!steps_name_.empty()) {
        auto steps_field = result_msg->get_field(steps_name_);
        if (steps_field) {
          auto steps_tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(steps_field);
          if (steps_tensor_msg) {
            const int64_t* steps_data = steps_tensor_msg->get_tensor().data_ptr<int64_t>();
            int64_t steps_size = steps_tensor_msg->get_tensor().numel();
            steps.assign(steps_data, steps_data + steps_size);
          }
        }
      }

      const int64_t* starts_data = starts.data_ptr<int64_t>();
      const int64_t* ends_data = ends.data_ptr<int64_t>();
      int64_t num_slices = starts.numel();

      // Default axes: [0, 1, 2, ...]
      if (axes.empty()) {
        for (int64_t i = 0; i < num_slices; ++i) {
          axes.push_back(i);
        }
      }

      // Default steps: all 1
      if (steps.empty()) {
        steps.assign(num_slices, 1);
      }

      const auto& in_shape = data.shape();
      std::vector<int64_t> out_shape = in_shape;
      std::vector<int64_t> slice_starts(in_shape.size(), 0);
      std::vector<int64_t> slice_ends = in_shape;
      std::vector<int64_t> slice_steps(in_shape.size(), 1);

      for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        if (axis < 0) axis += in_shape.size();

        int64_t start = starts_data[i];
        int64_t end = ends_data[i];
        int64_t step = steps[i];
        int64_t dim_size = in_shape[axis];

        // Handle negative indices
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;

        // Clamp to valid range based on step direction
        if (step > 0) {
          start = std::max(0L, std::min(start, dim_size));
          end = std::max(0L, std::min(end, dim_size));
        } else {
          // For negative step, start must be <= dim_size-1. end can be -1.
          start = std::max(0L, std::min(start, dim_size - 1));
          // If end was INT_MIN, clamp to -1
          end = std::max(-1L, std::min(end, dim_size - 1));
        }

        slice_starts[axis] = start;
        slice_ends[axis] = end;
        slice_steps[axis] = step;

        if (step > 0) {
          if (start < end) {
            out_shape[axis] = (end - start + step - 1) / step;
          } else {
            out_shape[axis] = 0;
          }
        } else {
          if (start > end) {
            out_shape[axis] = (start - end + (-step) - 1) / (-step);
          } else {
            out_shape[axis] = 0;
          }
        }
      }

      dynamic_tensor output(data.get_dtype(), out_shape);

      if (data.get_dtype() == dtype::float32) {
        slice_impl<float>(data, output, slice_starts, slice_ends, slice_steps);
      } else if (data.get_dtype() == dtype::float64) {
        slice_impl<double>(data, output, slice_starts, slice_ends, slice_steps);
      } else if (data.get_dtype() == dtype::int64) {
        slice_impl<int64_t>(data, output, slice_starts, slice_ends, slice_steps);
      } else if (data.get_dtype() == dtype::int32) {
        slice_impl<int32_t>(data, output, slice_starts, slice_ends, slice_steps);
      } else {
        throw std::runtime_error("slice: unsupported dtype");
      }

      log_node_output("slice", node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("slice [{}]: {}", node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }

 private:
  template <typename T>
  void slice_impl(const dynamic_tensor& data, dynamic_tensor& output,
                  const std::vector<int64_t>& slice_starts, const std::vector<int64_t>& slice_ends,
                  const std::vector<int64_t>& slice_steps) {
    const T* src = data.data_ptr<T>();
    T* dst = output.data_ptr<T>();
    const auto& in_shape = data.shape();
    const auto& out_shape = output.shape();

    std::function<void(int64_t, int64_t, int64_t)> copy_recursive;
    copy_recursive = [&](int64_t dim, int64_t src_offset, int64_t dst_offset) {
      if (dim == static_cast<int64_t>(in_shape.size())) {
        dst[dst_offset] = src[src_offset];
        return;
      }

      int64_t src_stride = 1;
      int64_t dst_stride = 1;
      for (int64_t i = dim + 1; i < static_cast<int64_t>(in_shape.size()); ++i) {
        src_stride *= in_shape[i];
      }
      for (int64_t i = dim + 1; i < static_cast<int64_t>(out_shape.size()); ++i) {
        dst_stride *= out_shape[i];
      }

      int64_t dst_idx = 0;
      int64_t start = slice_starts[dim];
      int64_t end = slice_ends[dim];
      int64_t step = slice_steps[dim];

      if (step > 0) {
        for (int64_t i = start; i < end; i += step) {
          copy_recursive(dim + 1, src_offset + i * src_stride, dst_offset + dst_idx * dst_stride);
          dst_idx++;
        }
      } else {
        for (int64_t i = start; i > end; i += step) {
          copy_recursive(dim + 1, src_offset + i * src_stride, dst_offset + dst_idx * dst_stride);
          dst_idx++;
        }
      }
    };

    copy_recursive(0, 0, 0);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::slice_node, coalsack::graph_node)
