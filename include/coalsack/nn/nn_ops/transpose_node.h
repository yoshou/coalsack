#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class transpose_node : public unary_op_node {
 private:
  std::vector<int64_t> perm_;

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& input_shape = input.shape();
    int64_t rank = input.ndim();

    // Use default perm if not set (reverse order)
    std::vector<int64_t> perm = perm_;
    if (perm.empty()) {
      perm.resize(rank);
      for (int64_t i = 0; i < rank; ++i) {
        perm[i] = rank - 1 - i;
      }
    }

    if (static_cast<int64_t>(perm.size()) != rank) {
      throw std::runtime_error("Perm size must match input rank");
    }

    std::vector<int64_t> output_shape(rank);

    for (int64_t i = 0; i < rank; ++i) {
      int64_t p = perm[i];
      if (p < 0) p += rank;
      if (p < 0 || p >= rank) {
        throw std::runtime_error("Invalid perm value");
      }
      output_shape[i] = input_shape[p];
    }

    dynamic_tensor output(input.get_dtype(), output_shape);

    auto input_strides = input.compute_strides();
    auto output_strides = output.compute_strides();

    int64_t numel = input.numel();

    if (input.get_dtype() == dtype::float32) {
      transpose_impl<float>(input, output, perm, input_strides, output_shape, numel, rank);
    } else if (input.get_dtype() == dtype::float64) {
      transpose_impl<double>(input, output, perm, input_strides, output_shape, numel, rank);
    } else if (input.get_dtype() == dtype::int64) {
      transpose_impl<int64_t>(input, output, perm, input_strides, output_shape, numel, rank);
    } else if (input.get_dtype() == dtype::int32) {
      transpose_impl<int32_t>(input, output, perm, input_strides, output_shape, numel, rank);
    } else {
      throw std::runtime_error("transpose: unsupported dtype");
    }

    return output;
  }

  template <typename T>
  void transpose_impl(const dynamic_tensor& input, dynamic_tensor& output,
                      const std::vector<int64_t>& perm, const std::vector<int64_t>& input_strides,
                      const std::vector<int64_t>& output_shape, int64_t numel, int64_t rank) const {
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();

    for (int64_t i = 0; i < numel; ++i) {
      std::vector<int64_t> indices(rank);
      int64_t temp = i;
      for (int64_t d = rank - 1; d >= 0; --d) {
        indices[d] = temp % output_shape[d];
        temp /= output_shape[d];
      }

      std::vector<int64_t> input_indices(rank);
      for (int64_t d = 0; d < rank; ++d) {
        input_indices[perm[d]] = indices[d];
      }

      int64_t input_idx = 0;
      for (int64_t d = 0; d < rank; ++d) {
        input_idx += input_indices[d] * input_strides[d];
      }

      output_data[i] = input_data[input_idx];
    }
  }

 public:
  transpose_node() : unary_op_node("transpose"), perm_() {}

  void set_perm(const std::vector<int64_t>& perm) { perm_ = perm; }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::transpose_node, coalsack::graph_node)
