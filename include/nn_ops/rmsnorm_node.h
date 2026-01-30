#pragma once

#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class rmsnorm_node : public binary_op_node {
 public:
  rmsnorm_node() : binary_op_node("rmsnorm"), epsilon_(1e-5f) {}

  void set_epsilon(float epsilon) { epsilon_ = epsilon; }

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& input, const dynamic_tensor& weight) {
    return compute(input, weight);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input, const dynamic_tensor& weight) override {
    // RMS(x) = sqrt(mean(x^2) + epsilon)
    // output = (x / RMS(x)) * weight

    // Input shape: [batch, seq_len, hidden_dim] or any shape with normalization on last dim
    // Weight shape: [hidden_dim]
    // Normalized over last dimension

    const auto& shape = input.shape();
    if (shape.empty()) {
      throw std::runtime_error("rmsnorm: input must have at least 1 dimension");
    }

    int64_t outer_size = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      outer_size *= shape[i];
    }
    int64_t hidden_dim = shape.back();

    // Check weight shape
    if (weight.ndim() != 1 || weight.dim(0) != hidden_dim) {
      throw std::runtime_error("rmsnorm: weight shape must be [hidden_dim]");
    }

    dynamic_tensor output(input.get_dtype(), shape);

    if (input.get_dtype() == dtype::float32) {
      compute_impl<float>(input, weight, output, outer_size, hidden_dim);
    } else if (input.get_dtype() == dtype::float64) {
      compute_impl<double>(input, weight, output, outer_size, hidden_dim);
    } else {
      throw std::runtime_error("rmsnorm: only float32 and float64 supported");
    }

    return output;
  }

 private:
  float epsilon_;

  template <typename T>
  void compute_impl(const dynamic_tensor& input, const dynamic_tensor& weight,
                    dynamic_tensor& output, int64_t outer_size, int64_t hidden_dim) {
    const T* x = input.data_ptr<T>();
    const T* w = weight.data_ptr<T>();
    T* out = output.data_ptr<T>();

    for (int64_t i = 0; i < outer_size; ++i) {
      // Compute RMS (root mean square)
      T sum_sq = 0;
      for (int64_t j = 0; j < hidden_dim; ++j) {
        T val = x[i * hidden_dim + j];
        sum_sq += val * val;
      }
      T rms = std::sqrt(sum_sq / hidden_dim + static_cast<T>(epsilon_));

      // Normalize and scale by weight
      for (int64_t j = 0; j < hidden_dim; ++j) {
        out[i * hidden_dim + j] = (x[i * hidden_dim + j] / rms) * w[j];
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::rmsnorm_node, coalsack::graph_node)
