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
// - 7 inputs: hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down
// - Weight format: 3D tensors [num_experts, expert_ffn_dim, hidden_dim] in FLOAT16
// - Bias format: 2D tensors [num_experts, expert_ffn_dim or hidden_dim] in FLOAT32
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
      throw std::runtime_error("expert_mlp: expected 8 inputs (hidden_states, weights, biases, selected_expert_ids), got " + 
                               std::to_string(inputs.size()));
    }

    const auto& hidden_states = inputs[0];
    const auto& w_up = inputs[1];
    const auto& w_gate = inputs[2];
    const auto& w_down = inputs[3];
    const auto& b_up = inputs[4];
    const auto& b_gate = inputs[5];
    const auto& b_down = inputs[6];
    const auto& selected_expert_ids = inputs[7];

    // Check if this expert is selected (early return optimization)
    if (!is_expert_selected(selected_expert_ids, expert_id_)) {
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

    // Validate weight shapes: all must be 3D
    // Expected: w_up/w_gate [num_experts, expert_ffn_dim, hidden_dim], w_down [num_experts, hidden_dim, expert_ffn_dim]
    if (w_up.ndim() != 3 || w_gate.ndim() != 3 || w_down.ndim() != 3) {
      throw std::runtime_error("expert_mlp: all weights must be 3D");
    }

    int64_t num_experts = w_up.dim(0);
    int64_t expert_ffn_dim = w_up.dim(1);

    // Validate weight dimensions match
    if (w_up.dim(0) != num_experts || w_up.dim(1) != expert_ffn_dim || w_up.dim(2) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w_up shape mismatch");
    }
    if (w_gate.dim(0) != num_experts || w_gate.dim(1) != expert_ffn_dim || w_gate.dim(2) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w_gate shape mismatch");
    }
    if (w_down.dim(0) != num_experts || w_down.dim(1) != hidden_dim || w_down.dim(2) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: w_down shape mismatch");
    }

    // Validate bias shapes: [num_experts, expert_ffn_dim] or [num_experts, hidden_dim]
    if (b_up.ndim() != 2 || b_up.dim(0) != num_experts || b_up.dim(1) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: b_up must be [num_experts=" + std::to_string(num_experts) + 
                              ", expert_ffn_dim=" + std::to_string(expert_ffn_dim) + "], got [" + 
                              std::to_string(b_up.dim(0)) + ", " + std::to_string(b_up.dim(1)) + "]");
    }
    if (b_gate.ndim() != 2 || b_gate.dim(0) != num_experts || b_gate.dim(1) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: b_gate must be [num_experts=" + std::to_string(num_experts) + 
                              ", expert_ffn_dim=" + std::to_string(expert_ffn_dim) + "], got [" + 
                              std::to_string(b_gate.dim(0)) + ", " + std::to_string(b_gate.dim(1)) + "]");
    }
    if (b_down.ndim() != 2 || b_down.dim(0) != num_experts || b_down.dim(1) != hidden_dim) {
      throw std::runtime_error("expert_mlp: b_down must be [num_experts=" + std::to_string(num_experts) + 
                              ", hidden_dim=" + std::to_string(hidden_dim) + "], got [" + 
                              std::to_string(b_down.dim(0)) + ", " + std::to_string(b_down.dim(1)) + "]");
    }

    // Validate expert_id
    if (expert_id_ < 0 || expert_id_ >= num_experts) {
      throw std::runtime_error("expert_mlp: expert_id " + std::to_string(expert_id_) + 
                               " out of range [0, " + std::to_string(num_experts) + ")");
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

    // Compute gated MLP with biases
    compute_impl(hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, output, 
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
    // Weights: [num_experts, expert_ffn_dim, hidden_dim]
    // Biases: [num_experts, expert_ffn_dim]
    // Access w[expert_id][out_idx][in_idx] via: expert_id * (expert_ffn_dim * hidden_dim) + out_idx * hidden_dim + in_idx
    // Access b[expert_id][out_idx] via: expert_id * expert_ffn_dim + out_idx
    for (int64_t out_idx = 0; out_idx < expert_ffn_dim; ++out_idx) {
      __m256 up_vec = _mm256_setzero_ps();
      __m256 gate_vec = _mm256_setzero_ps();
      
      int64_t in_idx = 0;
      // Process 8 elements at a time with AVX2
      for (; in_idx + 7 < hidden_dim; in_idx += 8) {
        // Weight index: w[expert_id_][out_idx][in_idx]
        int64_t weight_idx = expert_id_ * (expert_ffn_dim * hidden_dim) + out_idx * hidden_dim + in_idx;
        
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
        // Weight index: w[expert_id_][out_idx][in_idx]
        int64_t weight_idx = expert_id_ * (expert_ffn_dim * hidden_dim) + out_idx * hidden_dim + in_idx;
        float w_up_f32 = _cvtsh_ss(w_up_data[weight_idx]);
        float w_gate_f32 = _cvtsh_ss(w_gate_data[weight_idx]);
        up_sum += hidden_vec[in_idx] * w_up_f32;
        gate_sum += hidden_vec[in_idx] * w_gate_f32;
      }
      
      // Add biases: b[expert_id_][out_idx]
      int64_t bias_idx = expert_id_ * expert_ffn_dim + out_idx;
      const float up_v = up_sum + b_up_data[bias_idx];
      const float gate_v = gate_sum + b_gate_data[bias_idx];

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
    // Weights: [num_experts, hidden_dim, expert_ffn_dim]
    // Biases: [num_experts, hidden_dim]
    // Access w[expert_id][out_idx][in_idx] via: expert_id * (hidden_dim * expert_ffn_dim) + out_idx * expert_ffn_dim + in_idx
    // Access b[expert_id][out_idx] via: expert_id * hidden_dim + out_idx
    for (int64_t out_idx = 0; out_idx < hidden_dim; ++out_idx) {
      __m256 sum_vec = _mm256_setzero_ps();
      
      int64_t in_idx = 0;
      // Process 8 elements at a time with AVX2
      for (; in_idx + 7 < expert_ffn_dim; in_idx += 8) {
        // Weight index: w_down[expert_id_][out_idx][in_idx]
        int64_t weight_idx = expert_id_ * (hidden_dim * expert_ffn_dim) + out_idx * expert_ffn_dim + in_idx;
        
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
        // Weight index: w_down[expert_id_][out_idx][in_idx]
        int64_t weight_idx = expert_id_ * (hidden_dim * expert_ffn_dim) + out_idx * expert_ffn_dim + in_idx;
        float w_down_f32 = _cvtsh_ss(w_down_data[weight_idx]);
        sum += activated[in_idx] * w_down_f32;
      }
      
      // Add down projection bias: b[expert_id_][out_idx]
      int64_t bias_idx = expert_id_ * hidden_dim + out_idx;
      output_vec[out_idx] = sum + b_down_data[bias_idx];
    }
  }

  // Modified SwiGLU activation with clipping:
  //   gate_clipped = min(gate, limit)
  //   up_clipped = clamp(up, -limit, limit)
  //   activation = (gate_clipped / (1 + exp(alpha * (-gate_clipped)))) * (up_clipped + 1)
  // Followed by down projection: down(activation) + b_down
  void compute_impl(const dynamic_tensor& hidden_states,
                   const dynamic_tensor& w_up,
                   const dynamic_tensor& w_gate,
                   const dynamic_tensor& w_down,
                   const dynamic_tensor& b_up,
                   const dynamic_tensor& b_gate,
                   const dynamic_tensor& b_down,
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

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const float* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        float* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        compute_token(hidden_vec, w_up_data, w_gate_data, w_down_data,
                     b_up_data, b_gate_data, b_down_data, output_vec,
                     hidden_dim, expert_ffn_dim);


      }
    }
  }

  // Check if expert_id is in selected_expert_ids (O(top_k) linear search, typically ~4 elements)
  bool is_expert_selected(const dynamic_tensor& selected_ids, int expert_id) const {
    if (selected_ids.ndim() != 1) {
      throw std::runtime_error("expert_mlp: selected_expert_ids must be 1D, got " + 
                               std::to_string(selected_ids.ndim()) + "D");
    }

    if (selected_ids.get_dtype() != dtype::int32) {
      throw std::runtime_error("expert_mlp: selected_expert_ids must be int32");
    }

    const int32_t* ids = selected_ids.data_ptr<int32_t>();
    int64_t num_selected = selected_ids.dim(0);

    // O(top_k) linear search (top_k is typically 4, so this is very fast)
    for (int64_t i = 0; i < num_selected; ++i) {
      if (ids[i] == expert_id) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_mlp_node, coalsack::graph_node)
