#pragma once

#include <cmath>

#include "../nn_op_node.h"

namespace coalsack {

class sigmoid_node : public unary_op_node {
 public:
  sigmoid_node() : unary_op_node("sigmoid") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    dynamic_tensor output(input.get_dtype(), input.shape());

    if (input.get_dtype() == dtype::float32) {
      sigmoid_impl<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      sigmoid_impl<double>(input, output);
    } else {
      throw std::runtime_error("sigmoid: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void sigmoid_impl(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = T(1) / (T(1) + std::exp(-input_data[i]));
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::sigmoid_node, coalsack::graph_node)
