#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class non_zero_node : public unary_op_node {
 public:
  non_zero_node() : unary_op_node("non_zero") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() == dtype::float32) {
      return compute_impl<float>(input, [](float v) { return v != 0.0f; });
    } else if (input.get_dtype() == dtype::bool_) {
      return compute_impl<bool>(input, [](bool v) { return v; });
    } else if (input.get_dtype() == dtype::int64) {
      return compute_impl<int64_t>(input, [](int64_t v) { return v != 0; });
    }
    throw std::runtime_error("non_zero: unsupported dtype");
  }

 private:
  template <typename T, typename Pred>
  dynamic_tensor compute_impl(const dynamic_tensor& input, Pred is_nonzero) {
    const T* data = input.data_ptr<T>();
    int64_t count = 0;
    for (int64_t i = 0; i < input.numel(); ++i) {
      if (is_nonzero(data[i])) ++count;
    }

    int64_t rank = input.shape().size();
    dynamic_tensor output(dtype::int64, {rank, count});
    int64_t* out_data = output.data_ptr<int64_t>();

    int64_t idx = 0;
    for (int64_t i = 0; i < input.numel(); ++i) {
      if (is_nonzero(data[i])) {
        int64_t temp = i;
        for (int64_t d = rank - 1; d >= 0; --d) {
          out_data[d * count + idx] = temp % input.shape()[d];
          temp /= input.shape()[d];
        }
        ++idx;
      }
    }

    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::non_zero_node, coalsack::graph_node)
