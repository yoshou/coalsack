#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

// MoE Router node for Mixture-of-Experts (MoE) transformer models
// Performs expert selection and weight computation for token routing
//
// Architecture:
// - 3 inputs: hidden_states, gate_weights, gate_bias
// - Input format: hidden_states [batch, seq_len, hidden_dim] in FLOAT32/FLOAT64
//                 gate_weights [num_experts, hidden_dim] in FLOAT32/FLOAT64
//                 gate_bias [num_experts] in FLOAT32
// - Output format: [batch, seq_len, top_k, 2] where last dim is [expert_index, weight]
// - Routing strategy: Top-k selection with softmax normalization over selected experts
class moe_router_node : public variadic_op_node {
 public:
  moe_router_node() : variadic_op_node("moe_router", 3), num_experts_(0), top_k_(0) {}

  void set_config(int64_t num_experts, int64_t top_k) {
    num_experts_ = num_experts;
    top_k_ = top_k;
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& hidden_states,
                              const dynamic_tensor& gate_weights, const dynamic_tensor& gate_bias) {
    std::vector<dynamic_tensor> inputs = {hidden_states, gate_weights, gate_bias};
    return compute(inputs);
  }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    // Validate input count
    if (inputs.size() != 3) {
      throw std::runtime_error(
          "moe_router: expected 3 inputs (hidden_states, gate_weights, gate_bias), got " +
          std::to_string(inputs.size()));
    }

    const auto& hidden_states = inputs[0];
    const auto& gate_weights = inputs[1];
    const auto& gate_bias = inputs[2];

    // Validate hidden_states shape: [batch, seq_len, hidden_dim]
    if (hidden_states.ndim() != 3) {
      throw std::runtime_error(
          "moe_router: hidden_states must be 3D [batch, seq_len, hidden_dim], got " +
          std::to_string(hidden_states.ndim()) + "D");
    }

    // Validate gate_weights shape: [num_experts, hidden_dim] (after GGUF load)
    if (gate_weights.ndim() != 2) {
      throw std::runtime_error(
          "moe_router: gate_weights must be 2D [num_experts, hidden_dim], got " +
          std::to_string(gate_weights.ndim()) + "D");
    }

    // Validate gate_bias shape: [num_experts]
    if (gate_bias.ndim() != 1) {
      throw std::runtime_error("moe_router: gate_bias must be 1D [num_experts], got " +
                               std::to_string(gate_bias.ndim()) + "D");
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);

    // Validate gate_weights dimensions match
    // Shape is reversed: [num_experts, hidden_dim] after GGUF load
    if (gate_weights.dim(0) != num_experts_ || gate_weights.dim(1) != hidden_dim) {
      throw std::runtime_error("moe_router: gate_weights shape mismatch, expected [" +
                               std::to_string(num_experts_) + ", " + std::to_string(hidden_dim) +
                               "], got [" + std::to_string(gate_weights.dim(0)) + ", " +
                               std::to_string(gate_weights.dim(1)) + "]");
    }

    // Validate gate_bias dimensions
    if (gate_bias.dim(0) != num_experts_) {
      throw std::runtime_error(
          "moe_router: gate_bias must have num_experts=" + std::to_string(num_experts_) +
          " elements, got " + std::to_string(gate_bias.dim(0)));
    }

    // Allocate output: [batch, seq_len, top_k, 2] where last dim is [expert_index, weight]
    std::vector<int64_t> output_shape = {batch, seq_len, top_k_, 2};
    dynamic_tensor output(dtype::float32, output_shape);

    if (hidden_states.get_dtype() == dtype::float32) {
      compute_impl<float>(hidden_states, gate_weights, gate_bias, output, batch, seq_len,
                          hidden_dim);
    } else if (hidden_states.get_dtype() == dtype::float64) {
      compute_impl<double>(hidden_states, gate_weights, gate_bias, output, batch, seq_len,
                           hidden_dim);
    } else {
      throw std::runtime_error("moe_router: only float32 and float64 supported");
    }

    return output;
  }

 private:
  int64_t num_experts_;
  int64_t top_k_;

  // Expert routing implementation with top-k selection and softmax normalization
  //
  // Algorithm:
  // 1. Compute gating logits for all experts: logits = hidden_states @ gate_weights^T + gate_bias
  // 2. Select top-k experts with highest logit values (partial sort by descending score)
  // 3. Apply softmax normalization over the selected k logits only to produce routing weights
  //    - Uses numerically stable softmax: exp(logit - max) / sum(exp(logit - max))
  //    - Resulting weights sum to 1.0 across the k selected experts
  // 4. Output pairs of (expert_index, routing_weight) for each selected expert
  template <typename T>
  void compute_impl(const dynamic_tensor& hidden_states, const dynamic_tensor& gate_weights,
                    const dynamic_tensor& gate_bias, dynamic_tensor& output, int64_t batch,
                    int64_t seq_len, int64_t hidden_dim) {
    const T* hidden_data = hidden_states.data_ptr<T>();
    const T* gate_data = gate_weights.data_ptr<T>();
    const float* bias_data = gate_bias.data_ptr<float>();
    float* out_data = output.data_ptr<float>();  // Always float32 output

    // Allocate intermediate buffers
    std::vector<T> logits(num_experts_);
    std::vector<std::pair<T, int64_t>> scored_experts(num_experts_);
    std::vector<float> topk_weights(static_cast<size_t>(top_k_));

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const T* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;

        // Step 1: Compute gating logits with bias: (hidden_states @ gate_weights.T) + bias
        // gate_weights shape: [num_experts, hidden_dim]
        for (int64_t expert = 0; expert < num_experts_; ++expert) {
          T sum = 0;
          for (int64_t d = 0; d < hidden_dim; ++d) {
            sum += hidden_vec[d] * gate_data[expert * hidden_dim + d];
          }
          logits[expert] = sum + static_cast<T>(bias_data[expert]);
        }

        // Step 2: Create scored experts list for selection
        for (int64_t expert = 0; expert < num_experts_; ++expert) {
          scored_experts[expert] = {logits[expert], expert};
        }

        // Step 3: Select top-k experts by logit score (descending order)
        std::partial_sort(scored_experts.begin(), scored_experts.begin() + top_k_,
                          scored_experts.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Step 4: Apply softmax over selected top-k logits only
        // Find max for numerical stability
        T max_topk = scored_experts[0].first;
        for (int64_t k = 1; k < top_k_; ++k) {
          if (scored_experts[k].first > max_topk) {
            max_topk = scored_experts[k].first;
          }
        }

        // Compute exp(logit - max) and sum
        float sum_exp_topk = 0.0f;
        for (int64_t k = 0; k < top_k_; ++k) {
          const float v = static_cast<float>(scored_experts[k].first - max_topk);
          topk_weights[static_cast<size_t>(k)] = std::exp(v);
          sum_exp_topk += topk_weights[static_cast<size_t>(k)];
        }

        // Normalize to get softmax probabilities (sum to 1)
        const float inv_sum = (sum_exp_topk > 0.0f) ? (1.0f / sum_exp_topk) : 0.0f;
        for (int64_t k = 0; k < top_k_; ++k) {
          topk_weights[static_cast<size_t>(k)] *= inv_sum;
        }

        // Step 5: Write output: [expert_index, weight] for each top-k expert
        int64_t out_base = ((b * seq_len + s) * top_k_) * 2;
        for (int64_t k = 0; k < top_k_; ++k) {
          out_data[out_base + k * 2 + 0] =
              static_cast<float>(scored_experts[k].second);  // expert index
          out_data[out_base + k * 2 + 1] =
              topk_weights[static_cast<size_t>(k)];  // normalized weight
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::moe_router_node, coalsack::graph_node)
