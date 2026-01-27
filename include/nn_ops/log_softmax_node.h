#pragma once

#include <cmath>
#include <limits>

#include "../nn_op_node.h"

namespace coalsack {

class log_softmax_node : public unary_op_node {
 public:
  log_softmax_node() : unary_op_node("log_softmax"), axis_(-1) {}

  void set_axis(int64_t axis) { axis_ = axis; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& shape = input.shape();
    int64_t rank = shape.size();
    int64_t actual_axis = axis_ < 0 ? axis_ + rank : axis_;

    size_t outer_size = 1;
    for (int64_t i = 0; i < actual_axis; ++i) outer_size *= shape[i];
    size_t axis_size = shape[actual_axis];
    size_t inner_size = 1;
    for (size_t i = actual_axis + 1; i < shape.size(); ++i) inner_size *= shape[i];

    dynamic_tensor output(input.get_dtype(), shape);

    if (input.get_dtype() == dtype::float32) {
      compute_impl<float>(input, output, outer_size, axis_size, inner_size);
    } else if (input.get_dtype() == dtype::float64) {
      compute_impl<double>(input, output, outer_size, axis_size, inner_size);
    } else {
      throw std::runtime_error("log_softmax: unsupported dtype");
    }

    return output;
  }

 private:
  int64_t axis_;

  template <typename T>
  void compute_impl(const dynamic_tensor& input, dynamic_tensor& output, size_t outer_size,
                    size_t axis_size, size_t inner_size) {
    const T* in_data = input.data_ptr<T>();
    T* out_data = output.data_ptr<T>();

    for (size_t outer = 0; outer < outer_size; ++outer) {
      for (size_t inner = 0; inner < inner_size; ++inner) {
        T max_val = -std::numeric_limits<T>::infinity();
        for (size_t i = 0; i < axis_size; ++i) {
          max_val = std::max(max_val, in_data[(outer * axis_size + i) * inner_size + inner]);
        }

        T sum_exp = 0;
        for (size_t i = 0; i < axis_size; ++i) {
          sum_exp += std::exp(in_data[(outer * axis_size + i) * inner_size + inner] - max_val);
        }
        T log_sum_exp = std::log(sum_exp) + max_val;

        for (size_t i = 0; i < axis_size; ++i) {
          out_data[(outer * axis_size + i) * inner_size + inner] =
              in_data[(outer * axis_size + i) * inner_size + inner] - log_sum_exp;
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::log_softmax_node, coalsack::graph_node)
