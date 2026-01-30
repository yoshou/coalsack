#pragma once

#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class rope_node : public unary_op_node {
 public:
  rope_node() : unary_op_node("rope"), head_dim_(0), base_(10000.0f), scaling_factor_(1.0f) {}

  void set_config(int64_t head_dim, int64_t max_seq_len, float base = 10000.0f,
                  float scaling_factor = 1.0f) {
    head_dim_ = head_dim;
    base_ = base;
    scaling_factor_ = scaling_factor;
    precompute_freqs(max_seq_len);
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& input) { return compute(input); }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    // Input shape: [batch, num_heads, seq_len, head_dim]
    // Apply rotary position embedding to each position

    const auto& shape = input.shape();
    if (shape.size() != 4) {
      throw std::runtime_error("rope: input must have shape [batch, num_heads, seq_len, head_dim]");
    }

    int64_t batch = shape[0];
    int64_t num_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t head_dim = shape[3];

    if (head_dim != head_dim_) {
      throw std::runtime_error("rope: head_dim mismatch");
    }

    if (seq_len > static_cast<int64_t>(cos_cache_.size() / (head_dim / 2))) {
      throw std::runtime_error("rope: seq_len exceeds precomputed cache");
    }

    dynamic_tensor output(input.get_dtype(), shape);

    if (input.get_dtype() == dtype::float32) {
      compute_impl<float>(input, output, batch, num_heads, seq_len, head_dim);
    } else if (input.get_dtype() == dtype::float64) {
      compute_impl<double>(input, output, batch, num_heads, seq_len, head_dim);
    } else {
      throw std::runtime_error("rope: only float32 and float64 supported");
    }

    return output;
  }

 private:
  int64_t head_dim_;
  float base_;
  float scaling_factor_;
  std::vector<float> cos_cache_;
  std::vector<float> sin_cache_;

  void precompute_freqs(int64_t max_seq_len) {
    // YaRN scaling: freq = base^(-2i/head_dim) / scaling_factor
    // For standard RoPE: freq = base^(-2i/head_dim)
    int64_t half_dim = head_dim_ / 2;
    cos_cache_.resize(max_seq_len * half_dim);
    sin_cache_.resize(max_seq_len * half_dim);

    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
      for (int64_t i = 0; i < half_dim; ++i) {
        float freq = std::pow(base_, -2.0f * i / head_dim_) / scaling_factor_;
        float angle = pos * freq;
        cos_cache_[pos * half_dim + i] = std::cos(angle);
        sin_cache_[pos * half_dim + i] = std::sin(angle);
      }
    }
  }

  template <typename T>
  void compute_impl(const dynamic_tensor& input, dynamic_tensor& output, int64_t batch,
                    int64_t num_heads, int64_t seq_len, int64_t head_dim) {
    const T* x = input.data_ptr<T>();
    T* out = output.data_ptr<T>();

    int64_t half_dim = head_dim / 2;

    // Apply rotation for each position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t pos = 0; pos < seq_len; ++pos) {
          for (int64_t d = 0; d < half_dim; ++d) {
            // Index calculation
            int64_t base_idx = ((b * num_heads + h) * seq_len + pos) * head_dim;
            int64_t cache_idx = pos * half_dim + d;

            // Get pair of values to rotate
            T x0 = x[base_idx + d * 2];
            T x1 = x[base_idx + d * 2 + 1];

            // Get cos/sin from cache
            T cos_val = static_cast<T>(cos_cache_[cache_idx]);
            T sin_val = static_cast<T>(sin_cache_[cache_idx]);

            // Apply rotation: [cos -sin; sin cos] @ [x0; x1]
            out[base_idx + d * 2] = x0 * cos_val - x1 * sin_val;
            out[base_idx + d * 2 + 1] = x0 * sin_val + x1 * cos_val;
          }
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::rope_node, coalsack::graph_node)
