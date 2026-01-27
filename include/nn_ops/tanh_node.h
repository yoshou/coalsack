#pragma once

#include <cmath>

#include "../nn_op_node.h"

namespace coalsack {

class tanh_node : public unary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() == dtype::float32) {
      return compute_tanh<float>(input);
    } else if (input.get_dtype() == dtype::float64) {
      return compute_tanh<double>(input);
    } else {
      throw std::runtime_error("tanh: only float32 and float64 supported");
    }
  }

  template <typename T>
  dynamic_tensor compute_tanh(const dynamic_tensor& input) const {
    dynamic_tensor output(input.get_dtype(), input.shape());
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = std::tanh(input_data[i]);
    }

    return output;
  }

 public:
  tanh_node() : unary_op_node("tanh") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::tanh_node, coalsack::graph_node)
