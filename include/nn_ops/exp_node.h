#pragma once

#include <cmath>

#include "../nn_op_node.h"

namespace coalsack {

class exp_node : public unary_op_node {
 private:
  template <typename T>
  dynamic_tensor exp_impl(const dynamic_tensor& input) {
    dynamic_tensor output(input.get_dtype(), input.shape());
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = std::exp(input_data[i]);
    }

    return output;
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() == dtype::float32) {
      return exp_impl<float>(input);
    } else if (input.get_dtype() == dtype::float64) {
      return exp_impl<double>(input);
    } else {
      throw std::runtime_error("exp: unsupported dtype (only float32 and float64 supported)");
    }
  }

 public:
  exp_node() : unary_op_node("exp") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::exp_node, coalsack::graph_node)
