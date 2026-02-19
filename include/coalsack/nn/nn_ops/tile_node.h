#pragma once

#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class tile_node : public binary_op_node {
 public:
  tile_node() : binary_op_node("tile") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input,
                         const dynamic_tensor& repeats_tensor) override {
    const int64_t* repeats_data = repeats_tensor.data_ptr<int64_t>();
    std::vector<int64_t> repeats(repeats_data, repeats_data + repeats_tensor.numel());

    const auto& in_shape = input.shape();
    std::vector<int64_t> out_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      out_shape[i] = in_shape[i] * repeats[i];
    }

    dynamic_tensor output(input.get_dtype(), out_shape);

    if (input.get_dtype() == dtype::float32) {
      tile_copy<float>(input, output, repeats);
    } else if (input.get_dtype() == dtype::float64) {
      tile_copy<double>(input, output, repeats);
    } else if (input.get_dtype() == dtype::int64) {
      tile_copy<int64_t>(input, output, repeats);
    } else if (input.get_dtype() == dtype::int32) {
      tile_copy<int32_t>(input, output, repeats);
    } else {
      throw std::runtime_error("tile: unsupported dtype");
    }
    return output;
  }

 private:
  template <typename T>
  void tile_copy(const dynamic_tensor& input, dynamic_tensor& output,
                 const std::vector<int64_t>& repeats) {
    const T* in_data = input.data_ptr<T>();
    T* out_data = output.data_ptr<T>();
    const auto& in_shape = input.shape();
    const auto& out_shape = output.shape();

    for (int64_t i = 0; i < output.numel(); ++i) {
      int64_t in_idx = 0;
      int64_t temp = i;
      int64_t in_stride = 1;

      for (int64_t d = out_shape.size() - 1; d >= 0; --d) {
        int64_t out_coord = temp % out_shape[d];
        temp /= out_shape[d];
        int64_t in_coord = out_coord % in_shape[d];
        in_idx += in_coord * in_stride;
        in_stride *= in_shape[d];
      }
      out_data[i] = in_data[in_idx];
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::tile_node, coalsack::graph_node)
