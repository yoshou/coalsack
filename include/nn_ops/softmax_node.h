#pragma once

#include <algorithm>
#include <cmath>

#include "../nn_op_node.h"

namespace coalsack {

class softmax_node : public unary_op_node {
 public:
  softmax_node() : unary_op_node("softmax"), axis_(-1) {}

  void set_axis(int64_t axis) { axis_ = axis; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& shape = input.shape();
    int64_t rank = input.ndim();

    // Normalize axis
    int64_t axis = axis_;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      throw std::runtime_error("softmax: axis out of range");
    }

    dynamic_tensor output(input.get_dtype(), shape);

    if (input.get_dtype() == dtype::float32) {
      softmax_impl<float>(input, output, axis);
    } else if (input.get_dtype() == dtype::float64) {
      softmax_impl<double>(input, output, axis);
    } else {
      throw std::runtime_error("softmax: unsupported dtype");
    }

    return output;
  }

 private:
  int64_t axis_;

  template <typename T>
  void softmax_impl(const dynamic_tensor& input, dynamic_tensor& output, int64_t axis) {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();
    const auto& shape = input.shape();
    int64_t rank = input.ndim();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      outer_size *= shape[i];
    }

    int64_t axis_size = shape[axis];

    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
      inner_size *= shape[i];
    }

    // Apply softmax
    for (int64_t outer = 0; outer < outer_size; ++outer) {
      for (int64_t inner = 0; inner < inner_size; ++inner) {
        // Find max for numerical stability
        T max_val = -std::numeric_limits<T>::infinity();
        for (int64_t i = 0; i < axis_size; ++i) {
          int64_t idx = (outer * axis_size + i) * inner_size + inner;
          max_val = std::max(max_val, input_data[idx]);
        }

        T sum = T(0);
        for (int64_t i = 0; i < axis_size; ++i) {
          int64_t idx = (outer * axis_size + i) * inner_size + inner;
          T exp_val = std::exp(input_data[idx] - max_val);
          output_data[idx] = exp_val;
          sum += exp_val;
        }

        // Normalize
        for (int64_t i = 0; i < axis_size; ++i) {
          int64_t idx = (outer * axis_size + i) * inner_size + inner;
          output_data[idx] /= sum;
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::softmax_node, coalsack::graph_node)
