#pragma once

#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class scatter_nd_node : public variadic_op_node {
 public:
  scatter_nd_node() : variadic_op_node("scatter_nd", 3) {}

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];

    dynamic_tensor output(data.get_dtype(), data.shape());
    std::memcpy(output.data(), data.data(), data.bytes());

    if (data.get_dtype() == dtype::float32) {
      if (indices.get_dtype() == dtype::int32)
        scatter_nd_impl<float, int32_t>(output, indices, updates);
      else if (indices.get_dtype() == dtype::int64)
        scatter_nd_impl<float, int64_t>(output, indices, updates);
      else
        throw std::runtime_error("scatter_nd: unsupported indices dtype");
    } else if (data.get_dtype() == dtype::float64) {
      if (indices.get_dtype() == dtype::int32)
        scatter_nd_impl<double, int32_t>(output, indices, updates);
      else if (indices.get_dtype() == dtype::int64)
        scatter_nd_impl<double, int64_t>(output, indices, updates);
      else
        throw std::runtime_error("scatter_nd: unsupported indices dtype");
    } else if (data.get_dtype() == dtype::int32) {
      if (indices.get_dtype() == dtype::int32)
        scatter_nd_impl<int32_t, int32_t>(output, indices, updates);
      else if (indices.get_dtype() == dtype::int64)
        scatter_nd_impl<int32_t, int64_t>(output, indices, updates);
      else
        throw std::runtime_error("scatter_nd: unsupported indices dtype");
    } else if (data.get_dtype() == dtype::int64) {
      if (indices.get_dtype() == dtype::int32)
        scatter_nd_impl<int64_t, int32_t>(output, indices, updates);
      else if (indices.get_dtype() == dtype::int64)
        scatter_nd_impl<int64_t, int64_t>(output, indices, updates);
      else
        throw std::runtime_error("scatter_nd: unsupported indices dtype");
    } else {
      throw std::runtime_error("scatter_nd: unsupported data dtype");
    }

    return output;
  }

 private:
  template <typename T, typename IndexT>
  void scatter_nd_impl(dynamic_tensor& output, const dynamic_tensor& indices,
                       const dynamic_tensor& updates) {
    T* out_data = output.data_ptr<T>();
    const IndexT* idx_data = indices.data_ptr<IndexT>();
    const T* upd_data = updates.data_ptr<T>();

    const auto& idx_shape = indices.shape();
    int64_t num_indices = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(idx_shape.size()) - 1; ++i) {
      num_indices *= idx_shape[i];
    }
    int64_t idx_depth = idx_shape.back();

    int64_t slice_size = 1;
    for (size_t k = idx_depth; k < output.shape().size(); ++k) {
      slice_size *= output.shape()[k];
    }

    std::vector<int64_t> out_strides(output.shape().size());
    if (!out_strides.empty()) {
      out_strides.back() = 1;
      for (int64_t j = static_cast<int64_t>(out_strides.size()) - 2; j >= 0; --j) {
        out_strides[j] = out_strides[j + 1] * output.shape()[j + 1];
      }
    }

    for (int64_t i = 0; i < num_indices; ++i) {
      int64_t out_offset = 0;
      bool valid = true;
      for (int64_t d = 0; d < idx_depth; ++d) {
        int64_t coord = static_cast<int64_t>(idx_data[i * idx_depth + d]);
        if (coord < 0) coord += output.shape()[d];

        if (coord < 0 || coord >= output.shape()[d]) {
          valid = false;
          break;
        }
        out_offset += coord * out_strides[d];
      }

      if (valid) {
        for (int64_t k = 0; k < slice_size; ++k) {
          out_data[out_offset + k] = upd_data[i * slice_size + k];
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::scatter_nd_node, coalsack::graph_node)
