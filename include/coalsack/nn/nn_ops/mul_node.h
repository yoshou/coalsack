#pragma once

#include "coalsack/nn/nn_op_node.h"
#include "coalsack/nn/nn_ops/elementwise_helpers.h"

namespace coalsack {

class mul_node : public binary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() != b.get_dtype()) {
      throw std::runtime_error("mul: inputs must have same dtype");
    }

    if (a.get_dtype() == dtype::float32) {
      return elementwise_binary_op(
          a, b, dtype::float32,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<float>()[out_idx] =
                a.data_ptr<float>()[a_idx] * b.data_ptr<float>()[b_idx];
          });
    } else if (a.get_dtype() == dtype::float64) {
      return elementwise_binary_op(
          a, b, dtype::float64,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<double>()[out_idx] =
                a.data_ptr<double>()[a_idx] * b.data_ptr<double>()[b_idx];
          });
    } else if (a.get_dtype() == dtype::int32) {
      return elementwise_binary_op(
          a, b, dtype::int32,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<int32_t>()[out_idx] =
                a.data_ptr<int32_t>()[a_idx] * b.data_ptr<int32_t>()[b_idx];
          });
    } else if (a.get_dtype() == dtype::int64) {
      return elementwise_binary_op(
          a, b, dtype::int64,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<int64_t>()[out_idx] =
                a.data_ptr<int64_t>()[a_idx] * b.data_ptr<int64_t>()[b_idx];
          });
    } else {
      throw std::runtime_error("mul: unsupported dtype");
    }
  }

 public:
  mul_node() : binary_op_node("mul") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::mul_node, coalsack::graph_node)
