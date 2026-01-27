#pragma once

#include <cmath>

#include "../nn_op_node.h"

namespace coalsack {

class sqrt_node : public unary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() == dtype::float32) {
      return compute_sqrt<float>(input);
    } else if (input.get_dtype() == dtype::float64) {
      return compute_sqrt<double>(input);
    } else {
      throw std::runtime_error("sqrt: only float32 and float64 supported");
    }
  }

  template <typename T>
  dynamic_tensor compute_sqrt(const dynamic_tensor& input) const {
    dynamic_tensor output(input.get_dtype(), input.shape());
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = std::sqrt(input_data[i]);
    }

    return output;
  }

 public:
  sqrt_node() : unary_op_node("sqrt") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::sqrt_node, coalsack::graph_node)
