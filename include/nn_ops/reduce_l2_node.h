#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class reduce_l2_node : public unary_op_node {
 public:
  reduce_l2_node() : unary_op_node("reduce_l2"), keepdims_(true) {}

  void set_axes(const std::vector<int64_t>& axes) { axes_ = axes; }
  void set_keepdims(bool keepdims) { keepdims_ = keepdims; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& in_shape = input.shape();
    std::vector<int64_t> out_shape;

    // Normalize axes (if empty, reduce all axes)
    std::vector<int64_t> normalized_axes;
    if (axes_.empty()) {
      for (int64_t d = 0; d < static_cast<int64_t>(in_shape.size()); ++d) {
        normalized_axes.push_back(d);
      }
    } else {
      for (auto axis : axes_) {
        normalized_axes.push_back(axis < 0 ? in_shape.size() + axis : axis);
      }
    }

    for (int64_t d = 0; d < static_cast<int64_t>(in_shape.size()); ++d) {
      bool is_reduced =
          std::find(normalized_axes.begin(), normalized_axes.end(), d) != normalized_axes.end();
      if (is_reduced) {
        if (keepdims_) out_shape.push_back(1);
      } else {
        out_shape.push_back(in_shape[d]);
      }
    }

    if (out_shape.empty()) out_shape.push_back(1);

    dynamic_tensor output(input.get_dtype(), out_shape);

    if (input.get_dtype() == dtype::float32) {
      reduce_l2_impl<float>(input, output, normalized_axes);
    } else if (input.get_dtype() == dtype::float64) {
      reduce_l2_impl<double>(input, output, normalized_axes);
    } else if (input.get_dtype() == dtype::int64) {
      reduce_l2_impl<int64_t>(input, output, normalized_axes);
    } else if (input.get_dtype() == dtype::int32) {
      reduce_l2_impl<int32_t>(input, output, normalized_axes);
    } else {
      throw std::runtime_error("reduce_l2: unsupported dtype");
    }
    return output;
  }

 private:
  std::vector<int64_t> axes_;
  bool keepdims_;

  template <typename T>
  void reduce_l2_impl(const dynamic_tensor& input, dynamic_tensor& output,
                      const std::vector<int64_t>& axes) {
    const T* in_data = input.data_ptr<T>();
    T* out_data = output.data_ptr<T>();
    std::fill_n(out_data, output.numel(), T(0));

    const auto& in_shape = input.shape();
    const auto& out_shape = output.shape();

    for (int64_t i = 0; i < input.numel(); ++i) {
      std::vector<int64_t> coords(in_shape.size());
      int64_t temp = i;
      for (int64_t d = in_shape.size() - 1; d >= 0; --d) {
        coords[d] = temp % in_shape[d];
        temp /= in_shape[d];
      }

      // Map to output index
      std::vector<int64_t> out_coords;
      for (int64_t d = 0; d < static_cast<int64_t>(in_shape.size()); ++d) {
        bool is_reduced = std::find(axes.begin(), axes.end(), d) != axes.end();
        if (keepdims_) {
          out_coords.push_back(is_reduced ? 0 : coords[d]);
        } else if (!is_reduced) {
          out_coords.push_back(coords[d]);
        }
      }

      int64_t out_idx = 0;
      int64_t stride = 1;
      for (int64_t d = out_coords.size() - 1; d >= 0; --d) {
        out_idx += out_coords[d] * stride;
        stride *= out_shape[d];
      }

      out_data[out_idx] += in_data[i] * in_data[i];
    }

    // Take square root
    for (int64_t i = 0; i < output.numel(); ++i) {
      out_data[i] = std::sqrt(out_data[i]);
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::reduce_l2_node, coalsack::graph_node)
