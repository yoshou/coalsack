#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class not_node : public unary_op_node {
 public:
  not_node() : unary_op_node("not") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() != dtype::bool_) {
      throw std::runtime_error("not: input must be bool");
    }
    dynamic_tensor output(dtype::bool_, input.shape());
    const bool* in_data = input.data_ptr<bool>();
    bool* out_data = output.data_ptr<bool>();
    for (int64_t i = 0; i < input.numel(); ++i) {
      out_data[i] = !in_data[i];
    }
    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::not_node, coalsack::graph_node)
