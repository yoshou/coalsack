#pragma once

#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class layer_normalization_node : public variadic_op_node {
 public:
  layer_normalization_node()
      : variadic_op_node("layer_normalization", 3), axis_(-1), epsilon_(1e-5f) {}

  void set_axis(int64_t axis) { axis_ = axis; }
  void set_epsilon(float epsilon) { epsilon_ = epsilon; }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    const auto& X = inputs[0];
    const auto& scale = inputs[1];
    const auto& shape = X.shape();
    int64_t rank = shape.size();
    int64_t actual_axis = axis_ < 0 ? axis_ + rank : axis_;

    size_t outer_size = 1;
    for (int64_t i = 0; i < actual_axis; ++i) outer_size *= shape[i];
    size_t norm_size = 1;
    for (int64_t i = actual_axis; i < shape.size(); ++i) norm_size *= shape[i];

    dynamic_tensor output(X.get_dtype(), shape);

    if (X.get_dtype() == dtype::float32) {
      compute_impl<float>(X, scale, inputs, output, outer_size, norm_size);
    } else if (X.get_dtype() == dtype::float64) {
      compute_impl<double>(X, scale, inputs, output, outer_size, norm_size);
    } else {
      throw std::runtime_error("layer_normalization: unsupported dtype");
    }

    return output;
  }

 private:
  int64_t axis_;
  float epsilon_;

  template <typename T>
  void compute_impl(const dynamic_tensor& X, const dynamic_tensor& scale,
                    const std::vector<dynamic_tensor>& inputs, dynamic_tensor& output,
                    size_t outer_size, size_t norm_size) {
    const T* x_data = X.data_ptr<T>();
    const T* scale_data = scale.data_ptr<T>();
    const T* bias_data = inputs.size() > 2 ? inputs[2].data_ptr<T>() : nullptr;
    T* out_data = output.data_ptr<T>();

    for (int64_t outer = 0; outer < outer_size; ++outer) {
      T mean = 0;
      for (int64_t i = 0; i < norm_size; ++i) {
        mean += x_data[outer * norm_size + i];
      }
      mean /= norm_size;

      T variance = 0;
      for (int64_t i = 0; i < norm_size; ++i) {
        T diff = x_data[outer * norm_size + i] - mean;
        variance += diff * diff;
      }
      variance /= norm_size;

      T inv_std = static_cast<T>(1) / std::sqrt(variance + static_cast<T>(epsilon_));
      for (int64_t i = 0; i < norm_size; ++i) {
        size_t idx = outer * norm_size + i;
        T normalized = (x_data[idx] - mean) * inv_std;
        out_data[idx] = normalized * scale_data[i];
        if (bias_data) out_data[idx] += bias_data[i];
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::layer_normalization_node, coalsack::graph_node)
