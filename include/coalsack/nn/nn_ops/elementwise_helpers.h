#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

template <typename Func>
dynamic_tensor elementwise_binary_op(const dynamic_tensor& a, const dynamic_tensor& b,
                                     dtype output_dtype, Func op) {
  std::vector<int64_t> output_shape;
  try {
    output_shape = dynamic_tensor::broadcast_shape(a.shape(), b.shape());
  } catch (const std::exception& e) {
    std::string msg = "Broadcast failed: Shape A=[";
    auto shape_a = a.shape();
    for (size_t i = 0; i < shape_a.size(); ++i)
      msg += (i > 0 ? "," : "") + std::to_string(shape_a[i]);
    msg += "], Shape B=[";
    auto shape_b = b.shape();
    for (size_t i = 0; i < shape_b.size(); ++i)
      msg += (i > 0 ? "," : "") + std::to_string(shape_b[i]);
    msg += "]";
    throw std::runtime_error(msg);
  }

  dynamic_tensor output(output_dtype, output_shape);

  int64_t output_size = output.numel();
  auto a_strides = a.compute_strides();
  auto b_strides = b.compute_strides();

  for (int64_t i = 0; i < output_size; ++i) {
    std::vector<int64_t> indices(output_shape.size());
    int64_t temp = i;
    for (int64_t d = static_cast<int64_t>(output_shape.size()) - 1; d >= 0; --d) {
      indices[d] = temp % output_shape[d];
      temp /= output_shape[d];
    }

    int64_t a_idx = 0;
    int64_t b_idx = 0;

    int64_t a_offset = output_shape.size() - a.ndim();
    int64_t b_offset = output_shape.size() - b.ndim();

    for (size_t d = 0; d < output_shape.size(); ++d) {
      if (d >= static_cast<size_t>(a_offset)) {
        int64_t a_d = d - a_offset;
        int64_t a_dim_idx = (a.dim(a_d) == 1) ? 0 : indices[d];
        a_idx += a_dim_idx * a_strides[a_d];
      }

      if (d >= static_cast<size_t>(b_offset)) {
        int64_t b_d = d - b_offset;
        int64_t b_dim_idx = (b.dim(b_d) == 1) ? 0 : indices[d];
        b_idx += b_dim_idx * b_strides[b_d];
      }
    }

    op(a, b, output, a_idx, b_idx, i);
  }

  return output;
}

}  // namespace coalsack
