#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class greater_node : public binary_op_node {
 public:
  greater_node() : binary_op_node("greater") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    auto output_shape = dynamic_tensor::broadcast_shape(a.shape(), b.shape());
    dynamic_tensor output(dtype::bool_, output_shape);

    if (a.get_dtype() == dtype::float32) {
      greater_impl<float>(a, b, output, output_shape);
    } else if (a.get_dtype() == dtype::float64) {
      greater_impl<double>(a, b, output, output_shape);
    } else if (a.get_dtype() == dtype::int32) {
      greater_impl<int32_t>(a, b, output, output_shape);
    } else if (a.get_dtype() == dtype::int64) {
      greater_impl<int64_t>(a, b, output, output_shape);
    } else {
      throw std::runtime_error("greater: unsupported dtype");
    }
    return output;
  }

 private:
  template <typename T>
  void greater_impl(const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& output,
                    const std::vector<int64_t>& output_shape) {
    const T* a_data = a.data_ptr<T>();
    const T* b_data = b.data_ptr<T>();
    bool* out_data = output.data_ptr<bool>();

    for (size_t i = 0; i < output.numel(); ++i) {
      size_t a_idx = compute_broadcast_index(i, a.shape(), output_shape);
      size_t b_idx = compute_broadcast_index(i, b.shape(), output_shape);
      out_data[i] = a_data[a_idx] > b_data[b_idx];
    }
  }

  static size_t compute_broadcast_index(size_t linear_idx, const std::vector<int64_t>& src_shape,
                                        const std::vector<int64_t>& dst_shape) {
    size_t src_idx = 0;
    size_t src_stride = 1;
    size_t temp = linear_idx;

    for (int64_t d = dst_shape.size() - 1; d >= 0; --d) {
      size_t coord = temp % dst_shape[d];
      temp /= dst_shape[d];

      int64_t src_d =
          d - (static_cast<int64_t>(dst_shape.size()) - static_cast<int64_t>(src_shape.size()));
      if (src_d >= 0 && src_shape[src_d] > 1) {
        src_idx += coord * src_stride;
      }
      if (src_d >= 0 && src_d < static_cast<int64_t>(src_shape.size()) - 1) {
        src_stride *= src_shape[src_d + 1];
      }
    }
    return src_idx;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::greater_node, coalsack::graph_node)
