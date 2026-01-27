#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class or_node : public binary_op_node {
 public:
  or_node() : binary_op_node("or") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() != dtype::bool_ || b.get_dtype() != dtype::bool_) {
      throw std::runtime_error("or: inputs must be bool");
    }
    auto output_shape = dynamic_tensor::broadcast_shape(a.shape(), b.shape());
    dynamic_tensor output(dtype::bool_, output_shape);
    const bool* a_data = a.data_ptr<bool>();
    const bool* b_data = b.data_ptr<bool>();
    bool* out_data = output.data_ptr<bool>();

    for (int64_t i = 0; i < output.numel(); ++i) {
      int64_t a_idx = compute_broadcast_index(i, a.shape(), output_shape);
      int64_t b_idx = compute_broadcast_index(i, b.shape(), output_shape);
      out_data[i] = a_data[a_idx] || b_data[b_idx];
    }
    return output;
  }

 private:
  int64_t compute_broadcast_index(int64_t linear_idx, const std::vector<int64_t>& src_shape,
                                  const std::vector<int64_t>& dst_shape) {
    int64_t src_idx = 0;
    int64_t src_stride = 1;
    int64_t temp = linear_idx;
    for (int64_t d = dst_shape.size() - 1; d >= 0; --d) {
      int64_t coord = temp % dst_shape[d];
      temp /= dst_shape[d];
      if (d >= static_cast<int64_t>(dst_shape.size() - src_shape.size())) {
        int64_t src_d = d - (dst_shape.size() - src_shape.size());
        if (src_shape[src_d] > 1) {
          src_idx += coord * src_stride;
        }
        src_stride *= src_shape[src_d];
      }
    }
    return src_idx;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::or_node, coalsack::graph_node)
