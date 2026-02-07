#pragma once

#include <immintrin.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

// Expert MLP layer for Mixture-of-Experts (MoE) transformer models
// Implements gated MLP: down(activation(gate(x), up(x)))
// Uses modified SwiGLU activation with value clipping
// 
// Architecture:
// - 8 inputs: hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output
// - Weight format: 3D tensors [num_experts, expert_ffn_dim, hidden_dim] in FLOAT16
// - Bias format: 2D tensors [num_experts, expert_ffn_dim or hidden_dim] in FLOAT32
// - Router output: 4D tensor [batch, seq_len, top_k, 2] where last dim is [expert_id, weight]
class expert_mlp_node : public variadic_op_node {
 public:
  explicit expert_mlp_node(int expert_id = 0)
      : variadic_op_node("expert_mlp", 8), expert_id_(expert_id) {}

  int get_expert_id() const { return expert_id_; }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) { return compute(inputs); }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    // Validate input count
    if (inputs.size() != 8) {
      throw std::runtime_error("expert_mlp: expected 8 inputs (hidden_states, weights, biases, router_output), got " + 
                               std::to_string(inputs.size()));
    }

    const auto& hidden_states = inputs[0];
    const auto& w_up = inputs[1];
    const auto& w_gate = inputs[2];
    const auto& w_down = inputs[3];
    const auto& b_up = inputs[4];
    const auto& b_gate = inputs[5];
    const auto& b_down = inputs[6];
    const auto& router_output = inputs[7];

    // Validate router_output shape: [batch, seq_len, top_k, 2]
    if (router_output.ndim() != 4 || router_output.dim(3) != 2) {
      throw std::runtime_error("expert_mlp: router_output must be 4D [batch, seq_len, top_k, 2], got " + 
                               std::to_string(router_output.ndim()) + "D");
    }
    if (router_output.get_dtype() != dtype::float32) {
      throw std::runtime_error("expert_mlp: router_output must be float32");
    }

    // Check if this expert is selected by any token (early return optimization)
    if (!has_any_token_selecting_expert(router_output, expert_id_)) {
      // Return empty tensor to signal non-selection
      return dynamic_tensor(dtype::float32, {0});
    }

    // Validate hidden_states shape: [batch, seq_len, hidden_dim]
    if (hidden_states.ndim() != 3) {
      throw std::runtime_error("expert_mlp: hidden_states must be 3D [batch, seq_len, hidden_dim], got " + 
                               std::to_string(hidden_states.ndim()) + "D");
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);

    // Validate weight shapes: all must be 2D (single expert slices)
    // Expected: w_up/w_gate [expert_ffn_dim, hidden_dim], w_down [hidden_dim, expert_ffn_dim]
    if (w_up.ndim() != 2 || w_gate.ndim() != 2 || w_down.ndim() != 2) {
      throw std::runtime_error("expert_mlp: all weights must be 2D [expert_ffn_dim, hidden_dim] or [hidden_dim, expert_ffn_dim]");
    }

    int64_t expert_ffn_dim = w_up.dim(0);

    // Validate weight dimensions match
    if (w_up.dim(0) != expert_ffn_dim || w_up.dim(1) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w_up must be [" + std::to_string(expert_ffn_dim) + 
                              ", " + std::to_string(hidden_dim) + "], got [" + 
                              std::to_string(w_up.dim(0)) + ", " + std::to_string(w_up.dim(1)) + "]");
    }
    if (w_gate.dim(0) != expert_ffn_dim || w_gate.dim(1) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w_gate must be [" + std::to_string(expert_ffn_dim) + 
                              ", " + std::to_string(hidden_dim) + "], got [" + 
                              std::to_string(w_gate.dim(0)) + ", " + std::to_string(w_gate.dim(1)) + "]");
    }
    if (w_down.dim(0) != hidden_dim || w_down.dim(1) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: w_down must be [" + std::to_string(hidden_dim) + 
                              ", " + std::to_string(expert_ffn_dim) + "], got [" + 
                              std::to_string(w_down.dim(0)) + ", " + std::to_string(w_down.dim(1)) + "]");
    }

    // Validate bias shapes: [expert_ffn_dim] or [hidden_dim] (1D slices)
    if (b_up.ndim() != 1 || b_up.dim(0) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: b_up must be [expert_ffn_dim=" + std::to_string(expert_ffn_dim) + 
                              "], got [" + std::to_string(b_up.dim(0)) + "]");
    }
    if (b_gate.ndim() != 1 || b_gate.dim(0) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: b_gate must be [expert_ffn_dim=" + std::to_string(expert_ffn_dim) + 
                              "], got [" + std::to_string(b_gate.dim(0)) + "]");
    }
    if (b_down.ndim() != 1 || b_down.dim(0) != hidden_dim) {
      throw std::runtime_error("expert_mlp: b_down must be [hidden_dim=" + std::to_string(hidden_dim) + 
                              "], got [" + std::to_string(b_down.dim(0)) + "]");
    }

    // Validate dtype: weights must be FLOAT16
    if (w_up.get_dtype() != dtype::float16 || w_gate.get_dtype() != dtype::float16 || w_down.get_dtype() != dtype::float16) {
      throw std::runtime_error("expert_mlp: all weights must be FLOAT16");
    }

    // Validate hidden_states dtype
    if (hidden_states.get_dtype() != dtype::float32) {
      throw std::runtime_error("expert_mlp: hidden_states must be FLOAT32");
    }

    // Allocate output: same shape as hidden_states
    dynamic_tensor output(dtype::float32, hidden_states.shape());

    // Compute gated MLP with biases (with per-token conditional execution)
    compute_impl(hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output, output, 
                 batch, seq_len, hidden_dim, expert_ffn_dim);

    return output;
  }

 private:
  int expert_id_;

  // Convert fp16 → fp32 using F16C + AVX2 SIMD
  __attribute__((target("f16c,avx2,fma")))
  void compute_token(const float* hidden_vec,
                     const uint16_t* w_up_data,
                     const uint16_t* w_gate_data,
                     const uint16_t* w_down_data,
                     const float* b_up_data,
                     const float* b_gate_data,
                     const float* b_down_data,
                     float* output_vec,
                     int64_t hidden_dim,
                     int64_t expert_ffn_dim) const {
    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;

    // Allocate intermediate buffer for activated values
    std::vector<float> activated(expert_ffn_dim);

    // Step 1: Up/gate projections with biases, then modified SwiGLU activation
    // Weights: [expert_ffn_dim, hidden_dim]
    // Biases: [expert_ffn_dim]
    // Access w[out_idx][in_idx] via: out_idx * hidden_dim + in_idx
    // Access b[out_idx] via: out_idx
    for (int64_t out_idx = 0; out_idx < expert_ffn_dim; ++out_idx) {
      __m256 up_vec = _mm256_setzero_ps();
      __m256 gate_vec = _mm256_setzero_ps();
      
      int64_t in_idx = 0;
      // Process 8 elements at a time with AVX2
      for (; in_idx + 7 < hidden_dim; in_idx += 8) {
        // Weight index: w[out_idx][in_idx]
        int64_t weight_idx = out_idx * hidden_dim + in_idx;
        
        // Load 8 FP16 weights and convert to FP32
        __m128i w_up_fp16 = _mm_loadu_si128((__m128i*)(w_up_data + weight_idx));
        __m128i w_gate_fp16 = _mm_loadu_si128((__m128i*)(w_gate_data + weight_idx));
        __m256 w_up_fp32 = _mm256_cvtph_ps(w_up_fp16);
        __m256 w_gate_fp32 = _mm256_cvtph_ps(w_gate_fp16);
        
        // Load 8 hidden states
        __m256 hidden = _mm256_loadu_ps(hidden_vec + in_idx);
        
        // FMA: up_vec += hidden * w_up, gate_vec += hidden * w_gate
        up_vec = _mm256_fmadd_ps(hidden, w_up_fp32, up_vec);
        gate_vec = _mm256_fmadd_ps(hidden, w_gate_fp32, gate_vec);
      }
      
      // Horizontal sum of 8 elements
      float up_sum = up_vec[0] + up_vec[1] + up_vec[2] + up_vec[3] + 
                     up_vec[4] + up_vec[5] + up_vec[6] + up_vec[7];
      float gate_sum = gate_vec[0] + gate_vec[1] + gate_vec[2] + gate_vec[3] + 
                       gate_vec[4] + gate_vec[5] + gate_vec[6] + gate_vec[7];
      
      // Process remaining elements
      for (; in_idx < hidden_dim; ++in_idx) {
        // Weight index: w[out_idx][in_idx]
        int64_t weight_idx = out_idx * hidden_dim + in_idx;
        float w_up_f32 = _cvtsh_ss(w_up_data[weight_idx]);
        float w_gate_f32 = _cvtsh_ss(w_gate_data[weight_idx]);
        up_sum += hidden_vec[in_idx] * w_up_f32;
        gate_sum += hidden_vec[in_idx] * w_gate_f32;
      }
      
      // Add biases: b[out_idx]
      const float up_v = up_sum + b_up_data[out_idx];
      const float gate_v = gate_sum + b_gate_data[out_idx];

      // Modified SwiGLU activation with clipping:
      //   gate_clipped = min(gate_v, limit)
      //   up_clipped = clamp(up_v, -limit, limit)
      //   activation = (gate_clipped / (1 + exp(-alpha * gate_clipped))) * (up_clipped + 1)
      const float x = std::min(gate_v, limit);
      const float y = std::clamp(up_v, -limit, limit);
      const float out_glu = x / (1.0f + std::exp(alpha * (-x)));

      activated[out_idx] = out_glu * (y + 1.0f);
    }

    // Step 2: Down projection with bias
    // Weights: [hidden_dim, expert_ffn_dim]
    // Biases: [hidden_dim]
    // Access w[out_idx][in_idx] via: out_idx * expert_ffn_dim + in_idx
    // Access b[out_idx] via: out_idx
    for (int64_t out_idx = 0; out_idx < hidden_dim; ++out_idx) {
      __m256 sum_vec = _mm256_setzero_ps();
      
      int64_t in_idx = 0;
      // Process 8 elements at a time with AVX2
      for (; in_idx + 7 < expert_ffn_dim; in_idx += 8) {
        // Weight index: w_down[out_idx][in_idx]
        int64_t weight_idx = out_idx * expert_ffn_dim + in_idx;
        
        // Load 8 FP16 weights and convert to FP32
        __m128i w_down_fp16 = _mm_loadu_si128((__m128i*)(w_down_data + weight_idx));
        __m256 w_down_fp32 = _mm256_cvtph_ps(w_down_fp16);
        
        // Load 8 activated values
        __m256 act = _mm256_loadu_ps(activated.data() + in_idx);
        
        // FMA: sum_vec += act * w_down
        sum_vec = _mm256_fmadd_ps(act, w_down_fp32, sum_vec);
      }
      
      // Horizontal sum of 8 elements
      float sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3] + 
                  sum_vec[4] + sum_vec[5] + sum_vec[6] + sum_vec[7];
      
      // Process remaining elements
      for (; in_idx < expert_ffn_dim; ++in_idx) {
        // Weight index: w_down[out_idx][in_idx]
        int64_t weight_idx = out_idx * expert_ffn_dim + in_idx;
        float w_down_f32 = _cvtsh_ss(w_down_data[weight_idx]);
        sum += activated[in_idx] * w_down_f32;
      }
      
      // Add down projection bias: b[out_idx]
      output_vec[out_idx] = sum + b_down_data[out_idx];
    }
  }

  // Modified SwiGLU activation with clipping:
  //   gate_clipped = min(gate, limit)
  //   up_clipped = clamp(up, -limit, limit)
  //   activation = (gate_clipped / (1 + exp(alpha * (-gate_clipped)))) * (up_clipped + 1)
  // Followed by down projection: down(activation) + b_down
  // Token-level conditional execution: only compute for tokens that selected this expert
  void compute_impl(const dynamic_tensor& hidden_states,
                   const dynamic_tensor& w_up,
                   const dynamic_tensor& w_gate,
                   const dynamic_tensor& w_down,
                   const dynamic_tensor& b_up,
                   const dynamic_tensor& b_gate,
                   const dynamic_tensor& b_down,
                   const dynamic_tensor& router_output,
                   dynamic_tensor& output,
                   int64_t batch, int64_t seq_len,
                   int64_t hidden_dim, int64_t expert_ffn_dim) {
    const float* hidden_data = hidden_states.data_ptr<float>();
    const uint16_t* w_up_data = w_up.data_ptr<uint16_t>();
    const uint16_t* w_gate_data = w_gate.data_ptr<uint16_t>();
    const uint16_t* w_down_data = w_down.data_ptr<uint16_t>();
    const float* b_up_data = b_up.data_ptr<float>();
    const float* b_gate_data = b_gate.data_ptr<float>();
    const float* b_down_data = b_down.data_ptr<float>();
    float* out_data = output.data_ptr<float>();
    const float* router_data = router_output.data_ptr<float>();
    int64_t top_k = router_output.dim(2);

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const float* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        float* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        // Check if this token selected current expert_id_
        if (!is_token_selecting_expert(router_data, b, s, seq_len, top_k, expert_id_)) {
          // Zero out output for non-selected tokens
          std::memset(output_vec, 0, hidden_dim * sizeof(float));
          continue;
        }

        // Process selected token
        compute_token(hidden_vec, w_up_data, w_gate_data, w_down_data,
                     b_up_data, b_gate_data, b_down_data, output_vec,
                     hidden_dim, expert_ffn_dim);
      }
    }
  }

  // Check if expert_id is in selected_expert_ids (O(top_k) linear search, typically ~4 elements)
  // Check if a specific token selected this expert
  bool is_token_selecting_expert(const float* router_data, int64_t b, int64_t s, 
                                  int64_t seq_len, int64_t top_k, int expert_id) const {
    int64_t base_idx = (b * seq_len + s) * top_k * 2;
    for (int64_t k = 0; k < top_k; ++k) {
      if (static_cast<int>(router_data[base_idx + k * 2]) == expert_id) {
        return true;
      }
    }
    return false;
  }

  // Check if any token in the batch selected this expert (for early return)
  bool has_any_token_selecting_expert(const dynamic_tensor& router_output, int expert_id) const {
    const float* router_data = router_output.data_ptr<float>();
    int64_t batch = router_output.dim(0);
    int64_t seq_len = router_output.dim(1);
    int64_t top_k = router_output.dim(2);

    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        if (is_token_selecting_expert(router_data, b, s, seq_len, top_k, expert_id)) {
          return true;
        }
      }
    }
    return false;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_mlp_node, coalsack::graph_node)
