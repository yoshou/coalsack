#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// ONNX Floor op: element-wise floor of float tensors.
class floor_node : public unary_op_node {
 public:
  floor_node() : unary_op_node("floor") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    dynamic_tensor output(input.get_dtype(), input.shape());

    if (input.get_dtype() == dtype::float32) {
      floor_impl<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      floor_impl<double>(input, output);
    } else {
      throw std::runtime_error("floor: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void floor_impl(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* src = input.data_ptr<T>();
    T* dst = output.data_ptr<T>();
    for (int64_t i = 0; i < input.numel(); ++i) {
      dst[i] = std::floor(src[i]);
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::floor_node, coalsack::graph_node)
