#pragma once

#include <sstream>
#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class expand_node : public binary_op_node {
 public:
  expand_node() : binary_op_node("expand") {}

 protected:
  dynamic_tensor compute(const dynamic_tensor& input, const dynamic_tensor& shape_tensor) override {
    const int64_t* target_shape_data = shape_tensor.data_ptr<int64_t>();
    std::vector<int64_t> target_shape(target_shape_data, target_shape_data + shape_tensor.numel());

    const auto& input_shape = input.shape();

    if (target_shape.size() < input_shape.size()) {
      std::vector<int64_t> new_target(input_shape.size() - target_shape.size(), 1);
      new_target.insert(new_target.end(), target_shape.begin(), target_shape.end());
      target_shape = new_target;
    }

    for (size_t i = 0; i < target_shape.size(); ++i) {
      int64_t in_idx = i - (target_shape.size() - input_shape.size());

      if (in_idx >= 0) {
        int64_t in_dim = input_shape[in_idx];
        int64_t& out_dim = target_shape[i];

        if (out_dim == -1 || out_dim == 1) {
          out_dim = in_dim;
        } else if (in_dim != out_dim && in_dim != 1) {
          std::stringstream ss;
          ss << "expand: invalid broadcasting at dimension " << i << ". Input: " << in_dim
             << ", Target: " << out_dim;
          throw std::runtime_error(ss.str());
        }
      }
    }

    dynamic_tensor output(input.get_dtype(), target_shape);

    if (input.get_dtype() == dtype::float32) {
      broadcast_copy<float>(input, output);
    } else if (input.get_dtype() == dtype::float64) {
      broadcast_copy<double>(input, output);
    } else if (input.get_dtype() == dtype::int32) {
      broadcast_copy<int32_t>(input, output);
    } else if (input.get_dtype() == dtype::int64) {
      broadcast_copy<int64_t>(input, output);
    } else if (input.get_dtype() == dtype::bool_) {
      broadcast_copy<bool>(input, output);
    } else {
      throw std::runtime_error("expand: unsupported dtype");
    }
    return output;
  }

 private:
  template <typename T>
  void broadcast_copy(const dynamic_tensor& input, dynamic_tensor& output) {
    const T* in_data = input.data_ptr<T>();
    T* out_data = output.data_ptr<T>();
    const auto& in_shape = input.shape();
    const auto& out_shape = output.shape();

    for (int64_t i = 0; i < output.numel(); ++i) {
      int64_t current_idx = i;
      int64_t computed_in_idx = 0;
      int64_t accumulated_stride = 1;

      for (int64_t d = out_shape.size() - 1; d >= 0; --d) {
        int64_t out_dim_sz = out_shape[d];
        int64_t coord = current_idx % out_dim_sz;
        current_idx /= out_dim_sz;

        int64_t in_d =
            d - (static_cast<int64_t>(out_shape.size()) - static_cast<int64_t>(in_shape.size()));

        if (in_d >= 0) {
          int64_t in_dim_sz = in_shape[in_d];
          int64_t in_coord = 0;
          if (in_dim_sz > 1) {
            in_coord = coord;
          }
          computed_in_idx += in_coord * accumulated_stride;
          accumulated_stride *= in_dim_sz;
        }
      }
      out_data[i] = in_data[computed_in_idx];
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expand_node, coalsack::graph_node)
