#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class log_node : public unary_op_node {
 public:
  log_node() : unary_op_node("log") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    dynamic_tensor output(input.get_dtype(), input.shape());

    if (input.get_dtype() == dtype::float32) {
      log_impl<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      log_impl<double>(input, output);
    } else {
      throw std::runtime_error("log: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void log_impl(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = std::log(input_data[i]);
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::log_node, coalsack::graph_node)
