#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class sin_node : public unary_op_node {
 public:
  sin_node() : unary_op_node("sin") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    dynamic_tensor output(input.get_dtype(), input.shape());

    if (input.get_dtype() == dtype::float32) {
      sin_impl<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      sin_impl<double>(input, output);
    } else {
      throw std::runtime_error("sin: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void sin_impl(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();
    int64_t n = input.numel();
    for (int64_t i = 0; i < n; ++i) {
      output_data[i] = std::sin(input_data[i]);
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::sin_node, coalsack::graph_node)
