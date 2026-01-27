#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class relu_node : public unary_op_node {
 public:
  relu_node() : unary_op_node("relu") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    dynamic_tensor output(input.get_dtype(), input.shape());

    if (input.get_dtype() == dtype::float32) {
      relu_impl<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      relu_impl<double>(input, output);
    } else if (input.get_dtype() == dtype::int64) {
      relu_impl<int64_t>(input, output);
    } else if (input.get_dtype() == dtype::int32) {
      relu_impl<int32_t>(input, output);
    } else {
      throw std::runtime_error("relu: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void relu_impl(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = std::max(T(0), input_data[i]);
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::relu_node, coalsack::graph_node)
