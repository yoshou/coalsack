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
      : variadic_op_node("expert_mlp", 7), expert_id_(expert_id) {}

  int get_expert_id() const { return expert_id_; }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) { return compute(inputs); }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    // Validate input count
    if (inputs.size() != 7) {
      throw std::runtime_error("expert_mlp: expected 7 inputs (hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down), got " + 
                               std::to_string(inputs.size()));
    }

    const auto& hidden_states = inputs[0];
    const auto& w_up = inputs[1];
    const auto& w_gate = inputs[2];
    const auto& w_down = inputs[3];
    const auto& b_up = inputs[4];
    const auto& b_gate = inputs[5];
    const auto& b_down = inputs[6];

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

    // Convert fp16 weights to fp32
    dynamic_tensor w_up_fp32 = to_fp32(w_up);
    dynamic_tensor w_gate_fp32 = to_fp32(w_gate);
    dynamic_tensor w_down_fp32 = to_fp32(w_down);

    // Allocate output: same shape as hidden_states
    dynamic_tensor output(dtype::float32, hidden_states.shape());

    // Compute gated MLP with biases
    compute_impl(hidden_states, w_up_fp32, w_gate_fp32, w_down_fp32, b_up, b_gate, b_down, output, 
                 batch, seq_len, hidden_dim, expert_ffn_dim);

    return output;
  }

 private:
  int expert_id_;

  // Convert fp16 → fp32 using F16C intrinsics
  __attribute__((target("f16c")))
  static dynamic_tensor to_fp32(const dynamic_tensor& input) {
    dynamic_tensor output(dtype::float32, input.shape());

    const uint16_t* src = input.data_ptr<uint16_t>();
    float* dst = output.data_ptr<float>();
    int64_t numel = input.numel();

    int64_t i = 0;
    // Process 8 elements at a time with F16C
    for (; i + 7 < numel; i += 8) {
      __m128i h = _mm_loadu_si128((__m128i*)(src + i));
      __m256 f = _mm256_cvtph_ps(h);
      _mm256_storeu_ps(dst + i, f);
    }

    // Handle remaining elements
    for (; i < numel; ++i) {
      __m128i h = _mm_cvtsi32_si128(src[i]);
      __m128 f = _mm_cvtph_ps(h);
      dst[i] = _mm_cvtss_f32(f);
    }

    return output;
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
    const float* w_up_data = w_up.data_ptr<float>();
    const float* w_gate_data = w_gate.data_ptr<float>();
    const float* w_down_data = w_down.data_ptr<float>();
    const float* b_up_data = b_up.data_ptr<float>();
    const float* b_gate_data = b_gate.data_ptr<float>();
    const float* b_down_data = b_down.data_ptr<float>();
    float* out_data = output.data_ptr<float>();

    // Allocate intermediate buffer for activated values
    std::vector<float> activated(expert_ffn_dim);

    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const float* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        float* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        // Step 1: Up/gate projections with biases, then modified SwiGLU activation
        // Weights: [num_experts, expert_ffn_dim, hidden_dim]
        // Biases: [num_experts, expert_ffn_dim]
        // Access w[expert_id][out_idx][in_idx] via: expert_id * (expert_ffn_dim * hidden_dim) + out_idx * hidden_dim + in_idx
        // Access b[expert_id][out_idx] via: expert_id * expert_ffn_dim + out_idx
        for (int64_t out_idx = 0; out_idx < expert_ffn_dim; ++out_idx) {
          float up_sum = 0.0f;
          float gate_sum = 0.0f;
          
          for (int64_t in_idx = 0; in_idx < hidden_dim; ++in_idx) {
            // Weight index: w[expert_id_][out_idx][in_idx]
            int64_t weight_idx = expert_id_ * (expert_ffn_dim * hidden_dim) + out_idx * hidden_dim + in_idx;
            up_sum += hidden_vec[in_idx] * w_up_data[weight_idx];
            gate_sum += hidden_vec[in_idx] * w_gate_data[weight_idx];
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
          float sum = 0.0f;
          
          for (int64_t in_idx = 0; in_idx < expert_ffn_dim; ++in_idx) {
            // Weight index: w_down[expert_id_][out_idx][in_idx]
            int64_t weight_idx = expert_id_ * (hidden_dim * expert_ffn_dim) + out_idx * expert_ffn_dim + in_idx;
            sum += activated[in_idx] * w_down_data[weight_idx];
          }
          
          // Add down projection bias: b[expert_id_][out_idx]
          int64_t bias_idx = expert_id_ * hidden_dim + out_idx;
          output_vec[out_idx] = sum + b_down_data[bias_idx];
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_mlp_node, coalsack::graph_node)
