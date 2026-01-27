#pragma once

#include <cmath>

#include "../nn_op_node.h"
#include "elementwise_helpers.h"

namespace coalsack {

class pow_node : public binary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& base, const dynamic_tensor& exponent) override {
    if (base.get_dtype() != dtype::float32 || exponent.get_dtype() != dtype::float32) {
      throw std::runtime_error("pow: only float32 supported");
    }

    return elementwise_binary_op(
        base, exponent, dtype::float32,
        [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
           int64_t b_idx, int64_t out_idx) {
          out.data_ptr<float>()[out_idx] =
              std::pow(a.data_ptr<float>()[a_idx], b.data_ptr<float>()[b_idx]);
        });
  }

 public:
  pow_node() : binary_op_node("pow") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::pow_node, coalsack::graph_node)
