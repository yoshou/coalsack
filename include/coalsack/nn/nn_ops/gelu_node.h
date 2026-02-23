#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class gelu_node : public unary_op_node {
 public:
  gelu_node() : unary_op_node("gelu") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() != dtype::float32) {
      throw std::runtime_error("gelu: input must be float32");
    }

    dynamic_tensor output(dtype::float32, input.shape());
    const float* src = input.data_ptr<float>();
    float* dst = output.data_ptr<float>();
    int64_t n = input.numel();

    for (int64_t i = 0; i < n; ++i) {
      float x = src[i];
      dst[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }

    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::gelu_node, coalsack::graph_node)
