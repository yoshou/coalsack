#pragma once

#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class gather_elements_node : public binary_op_node {
 public:
  gather_elements_node() : binary_op_node("gather_elements"), axis_(0) {}

  void set_axis(int64_t axis) { axis_ = axis; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& data, const dynamic_tensor& indices) override {
    dynamic_tensor output(data.get_dtype(), indices.shape());

    if (data.get_dtype() == dtype::float32) {
      gather_elements_impl<float>(data, indices, output);
    } else if (data.get_dtype() == dtype::float64) {
      gather_elements_impl<double>(data, indices, output);
    } else if (data.get_dtype() == dtype::int32) {
      gather_elements_impl<int32_t>(data, indices, output);
    } else if (data.get_dtype() == dtype::int64) {
      gather_elements_impl<int64_t>(data, indices, output);
    } else {
      throw std::runtime_error("gather_elements: unsupported dtype");
    }
    return output;
  }

 private:
  int64_t axis_;

  template <typename T>
  void gather_elements_impl(const dynamic_tensor& data, const dynamic_tensor& indices,
                            dynamic_tensor& output) {
    const T* data_ptr = data.data_ptr<T>();
    const int64_t* indices_ptr = indices.data_ptr<int64_t>();
    T* out_ptr = output.data_ptr<T>();

    const auto& data_shape = data.shape();
    int64_t axis = axis_ < 0 ? data_shape.size() + axis_ : axis_;

    for (int64_t i = 0; i < indices.numel(); ++i) {
      std::vector<int64_t> coords(data_shape.size());
      int64_t temp = i;
      for (int64_t d = data_shape.size() - 1; d >= 0; --d) {
        coords[d] = temp % indices.shape()[d];
        temp /= indices.shape()[d];
      }

      int64_t idx = indices_ptr[i];
      if (idx < 0) idx += data_shape[axis];
      coords[axis] = idx;

      int64_t data_idx = 0;
      int64_t stride = 1;
      for (int64_t d = data_shape.size() - 1; d >= 0; --d) {
        data_idx += coords[d] * stride;
        stride *= data_shape[d];
      }

      out_ptr[i] = data_ptr[data_idx];
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::gather_elements_node, coalsack::graph_node)
