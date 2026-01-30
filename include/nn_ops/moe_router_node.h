#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class moe_router_node : public binary_op_node {
 public:
  moe_router_node() : binary_op_node("moe_router"), num_experts_(0), top_k_(0) {}

  void set_config(int64_t num_experts, int64_t top_k) {
    num_experts_ = num_experts;
    top_k_ = top_k;
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& hidden_states,
                              const dynamic_tensor& gate_weights) {
    return compute(hidden_states, gate_weights);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& hidden_states,
                        const dynamic_tensor& gate_weights) override {
    // hidden_states: [batch, seq_len, hidden_dim]
    // gate_weights: [hidden_dim, num_experts]
    // Output: [batch, seq_len, top_k, 2] where last dim is [expert_index, weight]

    if (hidden_states.ndim() != 3) {
      throw std::runtime_error("moe_router: hidden_states must have 3 dimensions");
    }

    if (gate_weights.ndim() != 2) {
      throw std::runtime_error("moe_router: gate_weights must have 2 dimensions");
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);

    if (gate_weights.dim(0) != hidden_dim || gate_weights.dim(1) != num_experts_) {
      throw std::runtime_error("moe_router: gate_weights shape mismatch");
    }

    // Output shape: [batch, seq_len, top_k, 2]
    std::vector<int64_t> output_shape = {batch, seq_len, top_k_, 2};
    dynamic_tensor output(dtype::float32, output_shape);

    if (hidden_states.get_dtype() == dtype::float32) {
      compute_impl<float>(hidden_states, gate_weights, output, batch, seq_len, hidden_dim);
    } else if (hidden_states.get_dtype() == dtype::float64) {
      compute_impl<double>(hidden_states, gate_weights, output, batch, seq_len, hidden_dim);
    } else {
      throw std::runtime_error("moe_router: only float32 and float64 supported");
    }

    return output;
  }

 private:
  int64_t num_experts_;
  int64_t top_k_;

  template <typename T>
  void compute_impl(const dynamic_tensor& hidden_states, const dynamic_tensor& gate_weights,
                    dynamic_tensor& output, int64_t batch, int64_t seq_len,
                    int64_t hidden_dim) {
    const T* hidden_data = hidden_states.data_ptr<T>();
    const T* gate_data = gate_weights.data_ptr<T>();
    float* out_data = output.data_ptr<float>();  // Always float32 output

    std::vector<T> logits(num_experts_);
    std::vector<std::pair<T, int64_t>> scored_experts(num_experts_);

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const T* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;

        // Compute gating logits: hidden_states @ gate_weights
        for (int64_t expert = 0; expert < num_experts_; ++expert) {
          T sum = 0;
          for (int64_t d = 0; d < hidden_dim; ++d) {
            sum += hidden_vec[d] * gate_data[d * num_experts_ + expert];
          }
          logits[expert] = sum;
        }

        // Apply softmax
        T max_logit = *std::max_element(logits.begin(), logits.end());
        T sum_exp = 0;
        for (int64_t expert = 0; expert < num_experts_; ++expert) {
          logits[expert] = std::exp(logits[expert] - max_logit);
          sum_exp += logits[expert];
        }
        for (int64_t expert = 0; expert < num_experts_; ++expert) {
          logits[expert] /= sum_exp;
        }

        // Create scored experts list
        for (int64_t expert = 0; expert < num_experts_; ++expert) {
          scored_experts[expert] = {logits[expert], expert};
        }

        // Sort by score descending (top-k)
        std::partial_sort(scored_experts.begin(), scored_experts.begin() + top_k_,
                         scored_experts.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        // Normalize top-k weights to sum to 1
        T top_k_sum = 0;
        for (int64_t k = 0; k < top_k_; ++k) {
          top_k_sum += scored_experts[k].first;
        }

        // Write output: [expert_index, weight]
        int64_t out_base = ((b * seq_len + s) * top_k_) * 2;
        for (int64_t k = 0; k < top_k_; ++k) {
          out_data[out_base + k * 2 + 0] = static_cast<float>(scored_experts[k].second);  // index
          out_data[out_base + k * 2 + 1] =
              static_cast<float>(scored_experts[k].first / top_k_sum);  // normalized weight
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::moe_router_node, coalsack::graph_node)
