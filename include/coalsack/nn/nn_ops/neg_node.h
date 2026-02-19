#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class neg_node : public unary_op_node {
 public:
  neg_node() : unary_op_node("neg") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    dynamic_tensor output(input.get_dtype(), input.shape());

    if (input.get_dtype() == dtype::float32) {
      neg_impl<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      neg_impl<double>(input, output);
    } else if (input.get_dtype() == dtype::int32) {
      neg_impl<int32_t>(input, output);
    } else if (input.get_dtype() == dtype::int64) {
      neg_impl<int64_t>(input, output);
    } else {
      throw std::runtime_error("neg: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void neg_impl(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      output_data[i] = -input_data[i];
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::neg_node, coalsack::graph_node)
