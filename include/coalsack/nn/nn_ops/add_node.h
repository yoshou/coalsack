#pragma once

#include "coalsack/nn/nn_op_node.h"
#include "coalsack/nn/nn_ops/elementwise_helpers.h"

namespace coalsack {

class add_node : public binary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() != b.get_dtype()) {
      throw std::runtime_error("add: input dtype mismatch");
    }

    dtype dt = a.get_dtype();
    if (dt == dtype::float32) {
      return elementwise_binary_op(
          a, b, dt,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<float>()[out_idx] =
                a.data_ptr<float>()[a_idx] + b.data_ptr<float>()[b_idx];
          });
    } else if (dt == dtype::float64) {
      return elementwise_binary_op(
          a, b, dt,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<double>()[out_idx] =
                a.data_ptr<double>()[a_idx] + b.data_ptr<double>()[b_idx];
          });
    } else if (dt == dtype::int32) {
      return elementwise_binary_op(
          a, b, dt,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<int32_t>()[out_idx] =
                a.data_ptr<int32_t>()[a_idx] + b.data_ptr<int32_t>()[b_idx];
          });
    } else if (dt == dtype::int64) {
      return elementwise_binary_op(
          a, b, dt,
          [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
             int64_t b_idx, int64_t out_idx) {
            out.data_ptr<int64_t>()[out_idx] =
                a.data_ptr<int64_t>()[a_idx] + b.data_ptr<int64_t>()[b_idx];
          });
    } else {
      throw std::runtime_error("add: unsupported dtype");
    }
  }

 public:
  add_node() : binary_op_node("add") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::add_node, coalsack::graph_node)
