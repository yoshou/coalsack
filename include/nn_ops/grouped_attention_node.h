#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class grouped_attention_node : public variadic_op_node {
 public:
  grouped_attention_node()
      : variadic_op_node("grouped_attention", 3),
        num_q_heads_(0),
        num_kv_heads_(0),
        head_dim_(0),
        scale_(1.0f) {}

  void set_config(int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim) {
    num_q_heads_ = num_q_heads;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (num_q_heads % num_kv_heads != 0) {
      throw std::runtime_error(
          "grouped_attention: num_q_heads must be divisible by num_kv_heads");
    }
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) {
    return compute(inputs);
  }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    if (inputs.size() != 3) {
      throw std::runtime_error("grouped_attention: expected 3 inputs (Q, K, V)");
    }

    const auto& query = inputs[0];
    const auto& key = inputs[1];
    const auto& value = inputs[2];

    // Expected shapes: [batch, seq_len, hidden_dim]
    if (query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3) {
      throw std::runtime_error("grouped_attention: inputs must have 3 dimensions");
    }

    int64_t batch = query.dim(0);
    int64_t seq_len = query.dim(1);
    int64_t hidden_dim = query.dim(2);

    // Verify dimensions
    if (hidden_dim != num_q_heads_ * head_dim_) {
      throw std::runtime_error("grouped_attention: query hidden_dim mismatch");
    }

    if (key.dim(0) != batch || key.dim(1) != seq_len ||
        key.dim(2) != num_kv_heads_ * head_dim_) {
      throw std::runtime_error("grouped_attention: key shape mismatch");
    }

    if (value.dim(0) != batch || value.dim(1) != seq_len ||
        value.dim(2) != num_kv_heads_ * head_dim_) {
      throw std::runtime_error("grouped_attention: value shape mismatch");
    }

    dynamic_tensor output(query.get_dtype(), query.shape());

    if (query.get_dtype() == dtype::float32) {
      compute_impl<float>(query, key, value, output, batch, seq_len);
    } else if (query.get_dtype() == dtype::float64) {
      compute_impl<double>(query, key, value, output, batch, seq_len);
    } else {
      throw std::runtime_error("grouped_attention: only float32 and float64 supported");
    }

    return output;
  }

 private:
  int64_t num_q_heads_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
  float scale_;

  template <typename T>
  void compute_impl(const dynamic_tensor& query, const dynamic_tensor& key,
                    const dynamic_tensor& value, dynamic_tensor& output, int64_t batch,
                    int64_t seq_len) {
    const T* q_data = query.data_ptr<T>();
    const T* k_data = key.data_ptr<T>();
    const T* v_data = value.data_ptr<T>();
    T* out_data = output.data_ptr<T>();

    int64_t group_size = num_q_heads_ / num_kv_heads_;

    // For each batch
    for (int64_t b = 0; b < batch; ++b) {
      // For each KV head group
      for (int64_t kv_head = 0; kv_head < num_kv_heads_; ++kv_head) {
        // Process all queries in this group
        for (int64_t group_idx = 0; group_idx < group_size; ++group_idx) {
          int64_t q_head = kv_head * group_size + group_idx;

          // Compute attention for this query head
          compute_single_head_attention(q_data, k_data, v_data, out_data, b, q_head, kv_head,
                                        seq_len);
        }
      }
    }
  }

  template <typename T>
  void compute_single_head_attention(const T* q_data, const T* k_data, const T* v_data,
                                     T* out_data, int64_t batch_idx, int64_t q_head_idx,
                                     int64_t kv_head_idx, int64_t seq_len) {
    // Compute scaled dot-product attention
    // scores = (Q @ K^T) * scale
    // attention = softmax(scores, causal_mask) @ V

    std::vector<T> scores(seq_len * seq_len);

    // Compute Q @ K^T
    for (int64_t i = 0; i < seq_len; ++i) {
      for (int64_t j = 0; j < seq_len; ++j) {
        T sum = 0;
        for (int64_t d = 0; d < head_dim_; ++d) {
          sum += get_q(q_data, batch_idx, q_head_idx, i, d, seq_len) *
                 get_k(k_data, batch_idx, kv_head_idx, j, d, seq_len);
        }
        scores[i * seq_len + j] = sum * static_cast<T>(scale_);
      }
    }

    // Apply causal mask (set future positions to -inf)
    for (int64_t i = 0; i < seq_len; ++i) {
      for (int64_t j = i + 1; j < seq_len; ++j) {
        scores[i * seq_len + j] = -std::numeric_limits<T>::infinity();
      }
    }

    // Softmax per row
    for (int64_t i = 0; i < seq_len; ++i) {
      // Find max for numerical stability
      T max_score = -std::numeric_limits<T>::infinity();
      for (int64_t j = 0; j <= i; ++j) {  // Only up to i due to causal mask
        max_score = std::max(max_score, scores[i * seq_len + j]);
      }

      // Exp and sum
      T sum_exp = 0;
      for (int64_t j = 0; j <= i; ++j) {
        scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_score);
        sum_exp += scores[i * seq_len + j];
      }

      // Normalize
      if (sum_exp > 0) {
        for (int64_t j = 0; j <= i; ++j) {
          scores[i * seq_len + j] /= sum_exp;
        }
      }
      // Future positions remain -inf (or 0 after exp)
      for (int64_t j = i + 1; j < seq_len; ++j) {
        scores[i * seq_len + j] = 0;
      }
    }

    // Output = attention @ V
    for (int64_t i = 0; i < seq_len; ++i) {
      for (int64_t d = 0; d < head_dim_; ++d) {
        T sum = 0;
        for (int64_t j = 0; j < seq_len; ++j) {
          sum += scores[i * seq_len + j] * get_v(v_data, batch_idx, kv_head_idx, j, d, seq_len);
        }
        set_output(out_data, batch_idx, q_head_idx, i, d, sum, seq_len);
      }
    }
  }

  // Helper functions to access Q/K/V with proper indexing
  template <typename T>
  T get_q(const T* q_data, int64_t batch, int64_t head, int64_t pos, int64_t dim,
          int64_t seq_len) const {
    // Q shape: [batch, seq_len, num_q_heads * head_dim]
    // Indexing: [batch][pos][head][dim]
    int64_t idx = (batch * seq_len + pos) * num_q_heads_ * head_dim_ + head * head_dim_ + dim;
    return q_data[idx];
  }

  template <typename T>
  T get_k(const T* k_data, int64_t batch, int64_t head, int64_t pos, int64_t dim,
          int64_t seq_len) const {
    // K shape: [batch, seq_len, num_kv_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_kv_heads_ * head_dim_ + head * head_dim_ + dim;
    return k_data[idx];
  }

  template <typename T>
  T get_v(const T* v_data, int64_t batch, int64_t head, int64_t pos, int64_t dim,
          int64_t seq_len) const {
    // V shape: [batch, seq_len, num_kv_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_kv_heads_ * head_dim_ + head * head_dim_ + dim;
    return v_data[idx];
  }

  template <typename T>
  void set_output(T* out_data, int64_t batch, int64_t head, int64_t pos, int64_t dim, T value,
                  int64_t seq_len) const {
    // Output shape: [batch, seq_len, num_q_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_q_heads_ * head_dim_ + head * head_dim_ + dim;
    out_data[idx] = value;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::grouped_attention_node, coalsack::graph_node)
