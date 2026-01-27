#pragma once

#include <stdexcept>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class where_node : public variadic_op_node {
 public:
  where_node() : variadic_op_node("where", 3) {}

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    const auto& condition = inputs[0];
    const auto& x = inputs[1];
    const auto& y = inputs[2];

    auto shape1 = dynamic_tensor::broadcast_shape(condition.shape(), x.shape());
    auto output_shape = dynamic_tensor::broadcast_shape(shape1, y.shape());
    dynamic_tensor output(x.get_dtype(), output_shape);

    if (x.get_dtype() == dtype::float32) {
      compute_impl<float>(condition, x, y, output);
    } else if (x.get_dtype() == dtype::float64) {
      compute_impl<double>(condition, x, y, output);
    } else if (x.get_dtype() == dtype::int64) {
      compute_impl<int64_t>(condition, x, y, output);
    } else if (x.get_dtype() == dtype::int32) {
      compute_impl<int32_t>(condition, x, y, output);
    } else {
      throw std::runtime_error("where_node: unsupported dtype");
    }

    return output;
  }

 private:
  template <typename T>
  void compute_impl(const dynamic_tensor& condition, const dynamic_tensor& x,
                    const dynamic_tensor& y, dynamic_tensor& output) {
    const bool* cond_data = condition.data_ptr<bool>();
    const T* x_data = x.data_ptr<T>();
    const T* y_data = y.data_ptr<T>();
    T* out_data = output.data_ptr<T>();

    auto out_shape = output.shape();

    for (size_t i = 0; i < output.numel(); ++i) {
      size_t cond_idx = compute_broadcast_index(i, condition.shape(), out_shape);
      size_t x_idx = compute_broadcast_index(i, x.shape(), out_shape);
      size_t y_idx = compute_broadcast_index(i, y.shape(), out_shape);
      out_data[i] = cond_data[cond_idx] ? x_data[x_idx] : y_data[y_idx];
    }
  }

  size_t compute_broadcast_index(size_t linear_idx, const std::vector<int64_t>& src_shape,
                                 const std::vector<int64_t>& dst_shape) {
    size_t src_idx = 0;
    size_t src_stride = 1;
    size_t temp = linear_idx;
    for (int64_t d = dst_shape.size() - 1; d >= 0; --d) {
      size_t coord = temp % dst_shape[d];
      temp /= dst_shape[d];

      if (d >= static_cast<int64_t>(dst_shape.size()) - static_cast<int64_t>(src_shape.size())) {
        size_t src_d = d - (dst_shape.size() - src_shape.size());

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

COALSACK_REGISTER_NODE(coalsack::where_node, coalsack::graph_node)
