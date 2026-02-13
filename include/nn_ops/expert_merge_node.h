#pragma once

#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class expert_merge_node : public variadic_op_node {
 public:
  expert_merge_node() : variadic_op_node("expert_merge", 33), num_experts_(0), top_k_(0) {}

  void set_config(int64_t num_experts, int64_t top_k) {
    num_experts_ = num_experts;
    top_k_ = top_k;
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) { return compute(inputs); }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    // inputs[0]: router output [batch, seq_len, top_k, 2]  (2 = [expert_index, weight])
    // inputs[1..32]: expert outputs [batch, seq_len, hidden_dim]
    // output: weighted sum [batch, seq_len, hidden_dim]

    if (inputs.size() != static_cast<size_t>(num_experts_ + 1)) {
      throw std::runtime_error("expert_merge: expected " + std::to_string(num_experts_ + 1) +
                               " inputs, got " + std::to_string(inputs.size()));
    }

    const auto& router_output = inputs[0];

    if (router_output.ndim() != 4 || router_output.dim(2) != top_k_ || router_output.dim(3) != 2) {
      throw std::runtime_error("expert_merge: router output must have shape [batch, seq_len, " +
                               std::to_string(top_k_) + ", 2]");
    }

    int64_t batch = router_output.dim(0);
    int64_t seq_len = router_output.dim(1);

    // Get hidden_dim from first non-empty expert output
    int64_t hidden_dim = 0;
    for (int i = 1; i <= num_experts_; ++i) {
      const auto& expert_out = inputs[i];
      if (expert_out.ndim() == 3 && expert_out.dim(0) > 0) {
        hidden_dim = expert_out.dim(2);
        break;
      }
    }

    if (hidden_dim == 0) {
      throw std::runtime_error("expert_merge: no non-empty expert outputs found");
    }

    // Verify all non-empty expert outputs have same shape
    for (int i = 1; i <= num_experts_; ++i) {
      const auto& expert_out = inputs[i];
      if (expert_out.ndim() == 3 && expert_out.dim(0) > 0) {
        if (expert_out.dim(0) != batch || expert_out.dim(1) != seq_len ||
            expert_out.dim(2) != hidden_dim) {
          throw std::runtime_error(
              "expert_merge: all non-empty expert outputs must have same shape");
        }
      }
    }

    // Output shape: [batch, seq_len, hidden_dim]
    std::vector<int64_t> output_shape = {batch, seq_len, hidden_dim};
    dynamic_tensor output(dtype::float32, output_shape);

    if (router_output.get_dtype() != dtype::float32) {
      throw std::runtime_error("expert_merge: only float32 supported");
    }

    const float* router_data = router_output.data_ptr<float>();
    float* out_data = output.data_ptr<float>();

    // Initialize output to zero
    std::memset(out_data, 0, output.numel() * sizeof(float));

    // For each token, compute weighted sum of top-k experts
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        int64_t router_base = ((b * seq_len + s) * top_k_) * 2;
        int64_t output_base = (b * seq_len + s) * hidden_dim;

        // Get top-k expert indices and weights
        for (int64_t k = 0; k < top_k_; ++k) {
          int expert_idx = static_cast<int>(router_data[router_base + k * 2 + 0]);
          float weight = router_data[router_base + k * 2 + 1];

          // Bounds check
          if (expert_idx < 0 || expert_idx >= num_experts_) {
            throw std::runtime_error("expert_merge: expert_idx out of range: " +
                                     std::to_string(expert_idx));
          }

          // Get expert output (inputs[1 + expert_idx])
          const auto& expert_output = inputs[1 + expert_idx];

          // Skip empty tensors (non-selected experts return [0])
          if (expert_output.ndim() == 0 || expert_output.dim(0) == 0) {
            continue;
          }

          const float* expert_data = expert_output.data_ptr<float>();
          int64_t expert_base = (b * seq_len + s) * hidden_dim;

          // Add weighted expert output to result
          for (int64_t d = 0; d < hidden_dim; ++d) {
            out_data[output_base + d] += weight * expert_data[expert_base + d];
          }
        }
      }
    }

    return output;
  }

 private:
  int64_t num_experts_;
  int64_t top_k_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_merge_node, coalsack::graph_node)
