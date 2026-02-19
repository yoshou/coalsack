#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class concat_node : public variadic_op_node {
 private:
  int64_t axis_;

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    if (inputs.empty()) {
      throw std::runtime_error("concat: requires at least one input");
    }

    dtype dt = inputs[0].get_dtype();
    for (const auto& input : inputs) {
      if (input.get_dtype() != dt) {
        throw std::runtime_error("concat: all inputs must have same dtype");
      }
    }

    int64_t rank = inputs[0].ndim();

    int64_t axis = axis_;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      throw std::runtime_error("concat: axis out of range");
    }

    const auto& ref_shape = inputs[0].shape();
    int64_t concat_dim_size = 0;

    for (const auto& input : inputs) {
      if (input.ndim() != rank) {
        throw std::runtime_error("concat: all inputs must have same rank");
      }

      for (int64_t i = 0; i < rank; ++i) {
        if (i != axis && input.dim(i) != ref_shape[i]) {
          throw std::runtime_error("concat: incompatible shapes");
        }
      }

      concat_dim_size += input.dim(axis);
    }

    std::vector<int64_t> output_shape = ref_shape;
    output_shape[axis] = concat_dim_size;

    dynamic_tensor output(dt, output_shape);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      outer_size *= ref_shape[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
      inner_size *= ref_shape[i];
    }

    auto concat_loop = [&](auto* dst_data, size_t elem_size) {
      std::vector<const void*> src_data;
      src_data.reserve(inputs.size());
      for (const auto& input : inputs) {
        src_data.push_back(input.data());
      }

      int64_t dst_offset = 0;
      for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
          int64_t axis_size = inputs[input_idx].dim(axis);
          int64_t copy_size = axis_size * inner_size;
          int64_t src_offset = outer * axis_size * inner_size;

          std::memcpy(reinterpret_cast<char*>(dst_data) + dst_offset * elem_size,
                      reinterpret_cast<const char*>(src_data[input_idx]) + src_offset * elem_size,
                      copy_size * elem_size);
          dst_offset += copy_size;
        }
      }
    };

    if (dt == dtype::float32) {
      concat_loop(output.data_ptr<float>(), sizeof(float));
    } else if (dt == dtype::float64) {
      concat_loop(output.data_ptr<double>(), sizeof(double));
    } else if (dt == dtype::int32) {
      concat_loop(output.data_ptr<int32_t>(), sizeof(int32_t));
    } else if (dt == dtype::int64) {
      concat_loop(output.data_ptr<int64_t>(), sizeof(int64_t));
    } else if (dt == dtype::bool_) {
      concat_loop(output.data_ptr<bool>(), sizeof(bool));
    } else {
      throw std::runtime_error("concat: unsupported dtype");
    }

    return output;
  }

 public:
  concat_node(size_t num_inputs = 2) : variadic_op_node("concat", num_inputs), axis_(0) {}

  void set_axis(int64_t axis) { axis_ = axis; }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::concat_node, coalsack::graph_node)
