#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class and_node : public binary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() != dtype::bool_ || b.get_dtype() != dtype::bool_) {
      throw std::runtime_error("and: both inputs must be bool");
    }

    return elementwise_binary_op(
        a, b, dtype::bool_,
        [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
           int64_t b_idx, int64_t out_idx) {
          out.data_ptr<bool>()[out_idx] = a.data_ptr<bool>()[a_idx] && b.data_ptr<bool>()[b_idx];
        });
  }

 public:
  and_node() : binary_op_node("and") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::and_node, coalsack::graph_node)
