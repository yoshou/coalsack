#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// Non-causal Scaled Dot-Product Attention for ViT.
// Inputs Q, K, V: [n_pos, n_embd] (n_embd = n_head * d_head)
// Output: [n_pos, n_embd]
class sdp_attention_node : public variadic_op_node {
 public:
  sdp_attention_node() : variadic_op_node("sdp_attention", 3), n_head_(0), d_head_(0) {}

  void set_num_heads(int n_head) { n_head_ = n_head; }
  void set_head_dim(int d_head) { d_head_ = d_head; }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    if (inputs.size() != 3) {
      throw std::runtime_error("sdp_attention: expected 3 inputs (Q, K, V)");
    }
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];

    if (Q.get_dtype() != dtype::float32 || K.get_dtype() != dtype::float32 ||
        V.get_dtype() != dtype::float32) {
      throw std::runtime_error("sdp_attention: all inputs must be float32");
    }
    if (Q.ndim() != 2 || K.ndim() != 2 || V.ndim() != 2) {
      throw std::runtime_error("sdp_attention: expected 2D inputs [n_pos, n_embd]");
    }
    if (n_head_ <= 0 || d_head_ <= 0) {
      throw std::runtime_error("sdp_attention: n_head and d_head must be set");
    }

    int64_t n_pos = Q.dim(0);
    int64_t n_embd = Q.dim(1);

    if (n_embd != static_cast<int64_t>(n_head_ * d_head_)) {
      throw std::runtime_error("sdp_attention: n_embd != n_head * d_head");
    }

    const float* q_data = Q.data_ptr<float>();
    const float* k_data = K.data_ptr<float>();
    const float* v_data = V.data_ptr<float>();

    dynamic_tensor output(dtype::float32, {n_pos, n_embd});
    float* out_data = output.data_ptr<float>();

    float scale = 1.0f / std::sqrt(static_cast<float>(d_head_));
    std::vector<float> scores(n_pos * n_pos);

    for (int h = 0; h < n_head_; ++h) {
      // Compute QK^T scores for this head
      for (int64_t i = 0; i < n_pos; ++i) {
        for (int64_t j = 0; j < n_pos; ++j) {
          double acc = 0.0;
          for (int d = 0; d < d_head_; ++d) {
            acc += static_cast<double>(q_data[i * n_embd + h * d_head_ + d]) *
                   static_cast<double>(k_data[j * n_embd + h * d_head_ + d]);
          }
          scores[i * n_pos + j] = static_cast<float>(acc) * scale;
        }
      }

      // Per-row softmax
      for (int64_t i = 0; i < n_pos; ++i) {
        float* row = scores.data() + i * n_pos;
        float max_s = *std::max_element(row, row + n_pos);
        float sum = 0.0f;
        for (int64_t j = 0; j < n_pos; ++j) {
          row[j] = std::exp(row[j] - max_s);
          sum += row[j];
        }
        for (int64_t j = 0; j < n_pos; ++j) {
          row[j] /= sum;
        }
      }

      // Weighted sum of V
      for (int64_t i = 0; i < n_pos; ++i) {
        const float* row = scores.data() + i * n_pos;
        for (int d = 0; d < d_head_; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < n_pos; ++j) {
            acc += static_cast<double>(row[j]) *
                   static_cast<double>(v_data[j * n_embd + h * d_head_ + d]);
          }
          out_data[i * n_embd + h * d_head_ + d] = static_cast<float>(acc);
        }
      }
    }

    return output;
  }

 private:
  int n_head_;
  int d_head_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::sdp_attention_node, coalsack::graph_node)
