#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "../gguf_dequant.h"
#include "../nn_op_node.h"

namespace coalsack {

// Grouped Query Attention (GQA) with optional sliding window and attention sinks.
// Implements multi-head attention where multiple query heads share the same key/value heads.
// Supports:
//   - Grouped query attention (GQA): num_q_heads > num_kv_heads
//   - Multi-query attention (MQA): num_kv_heads = 1
//   - Standard multi-head attention: num_q_heads = num_kv_heads
//   - Sliding window attention: restricts attention to recent tokens
//   - Attention sinks: StreamingLLM technique for long-context stability
class grouped_attention_node : public variadic_op_node {
 public:
  grouped_attention_node()
      : variadic_op_node("grouped_attention", 3),
        num_q_heads_(0),
        num_kv_heads_(0),
        head_dim_(0),
        scale_(1.0f),
        sliding_window_(0) {}

  // Configure attention parameters.
  // Args:
  //   num_q_heads: number of query heads
  //   num_kv_heads: number of key/value heads (must divide num_q_heads evenly)
  //   head_dim: dimension of each attention head
  //   sliding_window: if > 0, restrict attention to last N tokens (0 = full causal)
  //   attn_sinks: optional per-head sink logits [num_q_heads] for StreamingLLM
  void set_config(int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
                  int64_t sliding_window = 0,
                  std::optional<dynamic_tensor> attn_sinks = std::nullopt) {
    num_q_heads_ = num_q_heads;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    sliding_window_ = sliding_window;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (num_q_heads % num_kv_heads != 0) {
      throw std::runtime_error("grouped_attention: num_q_heads must be divisible by num_kv_heads");
    }

    if (attn_sinks.has_value()) {
      const auto& sinks = *attn_sinks;
      if (sinks.ndim() != 1) {
        throw std::runtime_error("grouped_attention: attn_sinks must be 1D tensor");
      }
      if (sinks.dim(0) != num_q_heads_) {
        throw std::runtime_error("grouped_attention: attn_sinks size mismatch (expected " +
                                 std::to_string(num_q_heads_) + ", got " +
                                 std::to_string(sinks.dim(0)) + ")");
      }
      attn_sinks_ = std::move(attn_sinks);
    }
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) { return compute(inputs); }

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

    if (key.dim(0) != batch || key.dim(1) != seq_len || key.dim(2) != num_kv_heads_ * head_dim_) {
      throw std::runtime_error("grouped_attention: key shape mismatch");
    }

    if (value.dim(0) != batch || value.dim(1) != seq_len ||
        value.dim(2) != num_kv_heads_ * head_dim_) {
      throw std::runtime_error("grouped_attention: value shape mismatch");
    }

    // ===== KV Cache Management =====
    const auto& key_new = key;
    const auto& value_new = value;

    dynamic_tensor key_to_use;
    dynamic_tensor value_to_use;
    int64_t effective_seq_len;

    if (k_cache_.ndim() == 0 || cached_seq_len_ == 0) {
      // Prefill phase: Initialize cache
      if (k_cache_.ndim() == 0) {
        // Cache not initialized - skip caching (for backward compatibility)
        key_to_use = key_new;
        value_to_use = value_new;
        effective_seq_len = seq_len;
      } else {
        // Cache initialized - populate it
        copy_to_cache(k_cache_, key_new, 0);
        copy_to_cache(v_cache_, value_new, 0);
        cached_seq_len_ = seq_len;

        // Use the cached portion
        key_to_use = slice_cache(k_cache_, cached_seq_len_);
        value_to_use = slice_cache(v_cache_, cached_seq_len_);
        effective_seq_len = cached_seq_len_;
      }
    } else {
      // Decode phase: Append to cache
      if (seq_len != 1) {
        throw std::runtime_error("grouped_attention: decode phase expects seq_len=1, got " +
                                 std::to_string(seq_len));
      }

      copy_to_cache(k_cache_, key_new, cached_seq_len_);
      copy_to_cache(v_cache_, value_new, cached_seq_len_);
      cached_seq_len_ += seq_len;

      // Use the full cached sequence
      key_to_use = slice_cache(k_cache_, cached_seq_len_);
      value_to_use = slice_cache(v_cache_, cached_seq_len_);
      effective_seq_len = cached_seq_len_;
    }

    dynamic_tensor output(query.get_dtype(), query.shape());

    if (query.get_dtype() == dtype::float32) {
      compute_impl<float>(query, key_to_use, value_to_use, output, batch, seq_len,
                          effective_seq_len);
    } else if (query.get_dtype() == dtype::float64) {
      compute_impl<double>(query, key_to_use, value_to_use, output, batch, seq_len,
                           effective_seq_len);
    } else {
      throw std::runtime_error("grouped_attention: only float32 and float64 supported");
    }

    return output;
  }

 public:
  // KV cache management
  void set_k_cache(const dynamic_tensor& cache) { k_cache_ = cache; }
  void set_v_cache(const dynamic_tensor& cache) { v_cache_ = cache; }
  void reset_cache() { cached_seq_len_ = 0; }
  int64_t get_cached_seq_len() const { return cached_seq_len_; }

 private:
  int64_t num_q_heads_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
  float scale_;
  int64_t sliding_window_;
  std::optional<dynamic_tensor> attn_sinks_;

  // KV cache state
  dynamic_tensor k_cache_;      // [batch, num_kv_heads, max_seq_len, head_dim]
  dynamic_tensor v_cache_;      // [batch, num_kv_heads, max_seq_len, head_dim]
  int64_t cached_seq_len_ = 0;  // Number of tokens currently cached

  template <typename T>
  void compute_impl(const dynamic_tensor& query, const dynamic_tensor& key,
                    const dynamic_tensor& value, dynamic_tensor& output, int64_t batch,
                    int64_t query_seq_len, int64_t key_seq_len) {
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
                                        query_seq_len, key_seq_len);
        }
      }
    }
  }

  // Helper functions for KV cache management
  void copy_to_cache(dynamic_tensor& cache, const dynamic_tensor& src, int64_t start_pos) {
    // cache: [batch, num_kv_heads, max_seq_len, head_dim]
    // src: [batch, seq_len, num_kv_heads * head_dim]
    if (src.get_dtype() != dtype::float32 || cache.get_dtype() != dtype::float32) {
      throw std::runtime_error("grouped_attention: cache only supports float32");
    }

    const float* src_data = src.data_ptr<float>();
    float* cache_data = cache.data_ptr<float>();

    int64_t batch = src.dim(0);
    int64_t seq_len = src.dim(1);
    int64_t max_seq_len = cache.dim(2);

    if (start_pos + seq_len > max_seq_len) {
      throw std::runtime_error("grouped_attention: cache overflow");
    }

    // Copy data: [batch, seq_len, num_kv_heads, head_dim] -> [batch, num_kv_heads,
    // start_pos:start_pos+seq_len, head_dim]
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t h = 0; h < num_kv_heads_; ++h) {
          for (int64_t d = 0; d < head_dim_; ++d) {
            int64_t src_idx = ((b * seq_len + s) * num_kv_heads_ + h) * head_dim_ + d;
            int64_t cache_idx =
                ((b * num_kv_heads_ + h) * max_seq_len + (start_pos + s)) * head_dim_ + d;
            cache_data[cache_idx] = src_data[src_idx];
          }
        }
      }
    }
  }

  dynamic_tensor slice_cache(const dynamic_tensor& cache, int64_t seq_len) {
    // cache: [batch, num_kv_heads, max_seq_len, head_dim]
    // output: [batch, seq_len, num_kv_heads * head_dim]
    int64_t batch = cache.dim(0);
    int64_t max_seq_len = cache.dim(2);

    if (seq_len > max_seq_len) {
      throw std::runtime_error("grouped_attention: slice exceeds cache size");
    }

    dynamic_tensor result(cache.get_dtype(), {batch, seq_len, num_kv_heads_ * head_dim_});
    const float* cache_data = cache.data_ptr<float>();
    float* result_data = result.data_ptr<float>();

    // Copy data: [batch, num_kv_heads, 0:seq_len, head_dim] -> [batch, seq_len, num_kv_heads,
    // head_dim]
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t h = 0; h < num_kv_heads_; ++h) {
          for (int64_t d = 0; d < head_dim_; ++d) {
            int64_t cache_idx = ((b * num_kv_heads_ + h) * max_seq_len + s) * head_dim_ + d;
            int64_t result_idx = ((b * seq_len + s) * num_kv_heads_ + h) * head_dim_ + d;
            result_data[result_idx] = cache_data[cache_idx];
          }
        }
      }
    }

    return result;
  }

  template <typename T>
  void compute_single_head_attention(const T* q_data, const T* k_data, const T* v_data, T* out_data,
                                     int64_t batch_idx, int64_t q_head_idx, int64_t kv_head_idx,
                                     int64_t query_seq_len, int64_t key_seq_len) {
    const int64_t window = sliding_window_ > 0 ? sliding_window_ : key_seq_len;
    const bool has_sinks = attn_sinks_.has_value();

    // Load sink logit once per head (dtype-aware conversion)
    T sink_logit = -std::numeric_limits<T>::infinity();
    if (has_sinks) {
      const auto& sinks = *attn_sinks_;
      if (sinks.get_dtype() == dtype::float32) {
        sink_logit = static_cast<T>(sinks.data_ptr<float>()[q_head_idx]);
      } else if (sinks.get_dtype() == dtype::float64) {
        sink_logit = static_cast<T>(sinks.data_ptr<double>()[q_head_idx]);
      } else if (sinks.get_dtype() == dtype::float16) {
        sink_logit = static_cast<T>(fp16_to_fp32(sinks.data_ptr<uint16_t>()[q_head_idx]));
      }
    }

    std::vector<T> attn_weights;

    // Process each query position (only the new tokens in query)
    for (int64_t i = 0; i < query_seq_len; ++i) {
      // Determine which key positions to attend to
      // In decode phase, i=0 (only 1 query token), attend to all 0:key_seq_len
      // In prefill phase, i=0..seq_len-1, attend to 0:i for each query
      int64_t query_pos = (key_seq_len - query_seq_len) + i;  // Absolute position in sequence
      const int64_t j1 = query_pos;
      const int64_t j0 = std::max<int64_t>(0, query_pos - window + 1);
      const int64_t span = j1 - j0 + 1;

      attn_weights.resize(span);

      // Compute attention scores: Q @ K^T / sqrt(head_dim)
      T max_score = -std::numeric_limits<T>::infinity();
      for (int64_t j = j0; j <= j1; ++j) {
        T sum = 0;
        for (int64_t d = 0; d < head_dim_; ++d) {
          sum += get_q(q_data, batch_idx, q_head_idx, i, d, query_seq_len) *
                 get_k(k_data, batch_idx, kv_head_idx, j, d, key_seq_len);
        }
        const T score = sum * static_cast<T>(scale_);
        attn_weights[j - j0] = score;
        max_score = std::max(max_score, score);
      }

      if (has_sinks) {
        max_score = std::max(max_score, sink_logit);
      }

      // Softmax with optional sink (numerically stable)
      T sum_exp = 0;
      for (int64_t jj = 0; jj < span; ++jj) {
        attn_weights[jj] = std::exp(attn_weights[jj] - max_score);
        sum_exp += attn_weights[jj];
      }

      if (has_sinks) {
        sum_exp += std::exp(sink_logit - max_score);
      }

      const T inv_sum = sum_exp > 0 ? T(1) / sum_exp : T(0);
      for (int64_t jj = 0; jj < span; ++jj) {
        attn_weights[jj] *= inv_sum;
      }

      // Weighted sum: output = softmax(scores) @ V
      for (int64_t d = 0; d < head_dim_; ++d) {
        T out = 0;
        for (int64_t j = j0; j <= j1; ++j) {
          out += attn_weights[j - j0] * get_v(v_data, batch_idx, kv_head_idx, j, d, key_seq_len);
        }
        set_output(out_data, batch_idx, q_head_idx, i, d, out, query_seq_len);
      }
    }
  }

  // Indexing helpers for packed Q/K/V tensors
  template <typename T>
  T get_q(const T* q_data, int64_t batch, int64_t head, int64_t pos, int64_t dim,
          int64_t seq_len) const {
    // Shape: [batch, seq_len, num_q_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_q_heads_ * head_dim_ + head * head_dim_ + dim;
    return q_data[idx];
  }

  template <typename T>
  T get_k(const T* k_data, int64_t batch, int64_t head, int64_t pos, int64_t dim,
          int64_t seq_len) const {
    // Shape: [batch, seq_len, num_kv_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_kv_heads_ * head_dim_ + head * head_dim_ + dim;
    return k_data[idx];
  }

  template <typename T>
  T get_v(const T* v_data, int64_t batch, int64_t head, int64_t pos, int64_t dim,
          int64_t seq_len) const {
    // Shape: [batch, seq_len, num_kv_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_kv_heads_ * head_dim_ + head * head_dim_ + dim;
    return v_data[idx];
  }

  template <typename T>
  void set_output(T* out_data, int64_t batch, int64_t head, int64_t pos, int64_t dim, T value,
                  int64_t seq_len) const {
    // Shape: [batch, seq_len, num_q_heads * head_dim]
    int64_t idx = (batch * seq_len + pos) * num_q_heads_ * head_dim_ + head * head_dim_ + dim;
    out_data[idx] = value;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::grouped_attention_node, coalsack::graph_node)
