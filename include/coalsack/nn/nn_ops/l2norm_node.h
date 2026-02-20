#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// L2 (RMS) normalization without a learned weight.
// output[i] = x[i] / sqrt(mean(x^2) + epsilon)
// Normalized over the last dimension.
class l2norm_node : public unary_op_node {
 public:
  l2norm_node() : unary_op_node("l2norm"), epsilon_(1e-5f) {}

  void set_epsilon(float epsilon) { epsilon_ = epsilon; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& shape = input.shape();
    if (shape.empty()) {
      throw std::runtime_error("l2norm: input must have at least 1 dimension");
    }

    int64_t outer_size = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) outer_size *= shape[i];
    int64_t hidden_dim = shape.back();

    dynamic_tensor output(input.get_dtype(), shape);

    if (input.get_dtype() == dtype::float32) {
      const float* x = input.data_ptr<float>();
      float* out = output.data_ptr<float>();
      for (int64_t i = 0; i < outer_size; ++i) {
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < hidden_dim; ++j) {
          float v = x[i * hidden_dim + j];
          sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / hidden_dim + epsilon_);
        float inv = 1.0f / rms;
        for (int64_t j = 0; j < hidden_dim; ++j) {
          out[i * hidden_dim + j] = x[i * hidden_dim + j] * inv;
        }
      }
    } else {
      throw std::runtime_error("l2norm: only float32 supported");
    }

    return output;
  }

 private:
  float epsilon_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::l2norm_node, coalsack::graph_node)
