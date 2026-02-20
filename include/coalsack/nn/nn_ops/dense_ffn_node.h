#pragma once

#include <immintrin.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// Dense FFN layer for use as a shared expert in MoE transformer models.
// Implements gated MLP with standard SwiGLU activation:
//   output = down_proj(silu(gate_proj(x)) * up_proj(x))
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Processes ALL tokens densely (no routing/selection).
//
// Architecture:
// - 4 inputs: hidden_states, gate_weight, up_weight, down_weight
// - Optional additional 3 inputs: gate_bias, up_bias, down_bias
// - Weight format: 2D tensors [ffn_dim, hidden_dim] or [hidden_dim, ffn_dim] in FLOAT16
// - Bias format: 1D tensors [ffn_dim] or [hidden_dim] in FLOAT32
// - Input hidden_states: [batch, seq_len, hidden_dim] in FLOAT32
// - Output: [batch, seq_len, hidden_dim] in FLOAT32
class dense_ffn_node : public variadic_op_node {
 public:
  dense_ffn_node() : variadic_op_node("dense_ffn", 4) {}

  std::string get_proc_name() const override { return "dense_ffn"; }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    if (inputs.size() != 4 && inputs.size() != 7) {
      throw std::runtime_error(
          "dense_ffn: expected 4 or 7 inputs (hidden, gate_w, up_w, down_w[, gate_b, up_b, "
          "down_b]), got " +
          std::to_string(inputs.size()));
    }

    const auto& hidden = inputs[0];
    const auto& gate_w = inputs[1];
    const auto& up_w = inputs[2];
    const auto& down_w = inputs[3];
    const bool has_bias = (inputs.size() == 7);

    if (hidden.ndim() != 3) {
      throw std::runtime_error("dense_ffn: hidden_states must be 3D [B, S, D], got " +
                               std::to_string(hidden.ndim()) + "D");
    }
    if (gate_w.ndim() != 2 || up_w.ndim() != 2 || down_w.ndim() != 2) {
      throw std::runtime_error("dense_ffn: weight tensors must be 2D");
    }
    const dtype weight_dtype = gate_w.get_dtype();
    if ((weight_dtype != dtype::float16 && weight_dtype != dtype::float32) ||
        up_w.get_dtype() != weight_dtype || down_w.get_dtype() != weight_dtype) {
      throw std::runtime_error("dense_ffn: weight tensors must all be FLOAT16 or all FLOAT32");
    }
    if (hidden.get_dtype() != dtype::float32) {
      throw std::runtime_error("dense_ffn: hidden_states must be FLOAT32");
    }

    const int64_t batch = hidden.dim(0);
    const int64_t seq_len = hidden.dim(1);
    const int64_t hidden_dim = hidden.dim(2);
    // gate_w/up_w: [ffn_dim, hidden_dim], down_w: [hidden_dim, ffn_dim]
    const int64_t ffn_dim = gate_w.dim(0);

    if (gate_w.dim(1) != hidden_dim || up_w.dim(0) != ffn_dim || up_w.dim(1) != hidden_dim) {
      throw std::runtime_error("dense_ffn: gate_w/up_w shape mismatch");
    }
    if (down_w.dim(0) != hidden_dim || down_w.dim(1) != ffn_dim) {
      throw std::runtime_error("dense_ffn: down_w shape mismatch, expected [" +
                               std::to_string(hidden_dim) + ", " + std::to_string(ffn_dim) +
                               "], got [" + std::to_string(down_w.dim(0)) + ", " +
                               std::to_string(down_w.dim(1)) + "]");
    }

    const float* gate_bias_ptr = has_bias ? inputs[4].data_ptr<float>() : nullptr;
    const float* up_bias_ptr = has_bias ? inputs[5].data_ptr<float>() : nullptr;
    const float* down_bias_ptr = has_bias ? inputs[6].data_ptr<float>() : nullptr;

    dynamic_tensor output(dtype::float32, {batch, seq_len, hidden_dim});
    float* out_ptr = output.data_ptr<float>();
    const float* hidden_ptr = hidden.data_ptr<float>();

    if (weight_dtype == dtype::float16) {
      const uint16_t* gate_w_data = gate_w.data_ptr<uint16_t>();
      const uint16_t* up_w_data = up_w.data_ptr<uint16_t>();
      const uint16_t* down_w_data = down_w.data_ptr<uint16_t>();
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          const float* hidden_vec = hidden_ptr + (b * seq_len + s) * hidden_dim;
          float* output_vec = out_ptr + (b * seq_len + s) * hidden_dim;
          compute_token(hidden_vec, gate_w_data, up_w_data, down_w_data, gate_bias_ptr, up_bias_ptr,
                        down_bias_ptr, output_vec, hidden_dim, ffn_dim);
        }
      }
    } else {
      const float* gate_w_data = gate_w.data_ptr<float>();
      const float* up_w_data = up_w.data_ptr<float>();
      const float* down_w_data = down_w.data_ptr<float>();
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          const float* hidden_vec = hidden_ptr + (b * seq_len + s) * hidden_dim;
          float* output_vec = out_ptr + (b * seq_len + s) * hidden_dim;
          compute_token_f32(hidden_vec, gate_w_data, up_w_data, down_w_data, gate_bias_ptr,
                            up_bias_ptr, down_bias_ptr, output_vec, hidden_dim, ffn_dim);
        }
      }
    }

    return output;
  }

 private:
  // F32 weight compute path (scalar)
  void compute_token_f32(const float* hidden_vec, const float* gate_w_data, const float* up_w_data,
                         const float* down_w_data, const float* gate_bias_ptr,
                         const float* up_bias_ptr, const float* down_bias_ptr, float* output_vec,
                         int64_t hidden_dim, int64_t ffn_dim) const {
    std::vector<float> activated(static_cast<size_t>(ffn_dim));

    for (int64_t out_idx = 0; out_idx < ffn_dim; ++out_idx) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (int64_t in_idx = 0; in_idx < hidden_dim; ++in_idx) {
        const int64_t w_idx = out_idx * hidden_dim + in_idx;
        gate_sum += hidden_vec[in_idx] * gate_w_data[w_idx];
        up_sum += hidden_vec[in_idx] * up_w_data[w_idx];
      }
      if (gate_bias_ptr) gate_sum += gate_bias_ptr[out_idx];
      if (up_bias_ptr) up_sum += up_bias_ptr[out_idx];
      const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
      activated[static_cast<size_t>(out_idx)] = silu_gate * up_sum;
    }

    for (int64_t out_idx = 0; out_idx < hidden_dim; ++out_idx) {
      float sum = 0.0f;
      for (int64_t in_idx = 0; in_idx < ffn_dim; ++in_idx) {
        sum += activated[static_cast<size_t>(in_idx)] * down_w_data[out_idx * ffn_dim + in_idx];
      }
      if (down_bias_ptr) sum += down_bias_ptr[out_idx];
      output_vec[out_idx] = sum;
    }
  }

  // Horizontal sum of an AVX2 register
  __attribute__((target("avx2"))) static float horizontal_sum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
  }

  // FP16 compute with standard SwiGLU activation: silu(gate) * up
  __attribute__((target("f16c,avx2,fma"))) void compute_token(
      const float* hidden_vec, const uint16_t* gate_w_data, const uint16_t* up_w_data,
      const uint16_t* down_w_data, const float* gate_bias_ptr, const float* up_bias_ptr,
      const float* down_bias_ptr, float* output_vec, int64_t hidden_dim, int64_t ffn_dim) const {
    std::vector<float> activated(static_cast<size_t>(ffn_dim));

    // Step 1: Gate + Up projections, then SwiGLU activation
    for (int64_t out_idx = 0; out_idx < ffn_dim; ++out_idx) {
      __m256 gate_acc = _mm256_setzero_ps();
      __m256 up_acc = _mm256_setzero_ps();
      int64_t in_idx = 0;
      for (; in_idx + 7 < hidden_dim; in_idx += 8) {
        const int64_t w_idx = out_idx * hidden_dim + in_idx;
        __m128i gate_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gate_w_data + w_idx));
        __m128i up_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(up_w_data + w_idx));
        __m256 gate_fp32 = _mm256_cvtph_ps(gate_fp16);
        __m256 up_fp32 = _mm256_cvtph_ps(up_fp16);
        __m256 h = _mm256_loadu_ps(hidden_vec + in_idx);
        gate_acc = _mm256_fmadd_ps(h, gate_fp32, gate_acc);
        up_acc = _mm256_fmadd_ps(h, up_fp32, up_acc);
      }
      float gate_sum = horizontal_sum_avx2(gate_acc);
      float up_sum = horizontal_sum_avx2(up_acc);
      for (; in_idx < hidden_dim; ++in_idx) {
        const int64_t w_idx = out_idx * hidden_dim + in_idx;
        gate_sum += hidden_vec[in_idx] * _cvtsh_ss(gate_w_data[w_idx]);
        up_sum += hidden_vec[in_idx] * _cvtsh_ss(up_w_data[w_idx]);
      }
      if (gate_bias_ptr) gate_sum += gate_bias_ptr[out_idx];
      if (up_bias_ptr) up_sum += up_bias_ptr[out_idx];

      // Standard SwiGLU: silu(gate) * up
      const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
      activated[static_cast<size_t>(out_idx)] = silu_gate * up_sum;
    }

    // Step 2: Down projection
    for (int64_t out_idx = 0; out_idx < hidden_dim; ++out_idx) {
      __m256 sum_acc = _mm256_setzero_ps();
      int64_t in_idx = 0;
      for (; in_idx + 7 < ffn_dim; in_idx += 8) {
        const int64_t w_idx = out_idx * ffn_dim + in_idx;
        __m128i down_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(down_w_data + w_idx));
        __m256 down_fp32 = _mm256_cvtph_ps(down_fp16);
        __m256 act = _mm256_loadu_ps(activated.data() + in_idx);
        sum_acc = _mm256_fmadd_ps(act, down_fp32, sum_acc);
      }
      float sum = horizontal_sum_avx2(sum_acc);
      for (; in_idx < ffn_dim; ++in_idx) {
        const int64_t w_idx = out_idx * ffn_dim + in_idx;
        sum += activated[static_cast<size_t>(in_idx)] * _cvtsh_ss(down_w_data[w_idx]);
      }
      if (down_bias_ptr) sum += down_bias_ptr[out_idx];
      output_vec[out_idx] = sum;
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::dense_ffn_node, coalsack::graph_node)
