#pragma once

#include <unordered_set>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

// Expert Selector node for Mixture-of-Experts (MoE) optimization
// Extracts unique expert IDs from router output to enable conditional expert execution
// 
// Architecture:
// - 1 input: router_output [batch, seq_len, top_k, 2] where last dim is [expert_index, weight]
// - Output: selected_expert_ids [num_selected] containing unique expert IDs (int32)
// - Purpose: Parse router output once to avoid redundant checks in 32 expert nodes
class expert_selector_node : public unary_op_node {
 public:
  expert_selector_node() : unary_op_node("expert_selector") {}

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& router_output) {
    return compute(router_output);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& router_output) override {
    // Validate router_output shape: [batch, seq_len, top_k, 2]
    if (router_output.ndim() != 4 || router_output.dim(3) != 2) {
      throw std::runtime_error("expert_selector: router_output must be 4D [batch, seq_len, top_k, 2], got " + 
                               std::to_string(router_output.ndim()) + "D");
    }

    if (router_output.get_dtype() != dtype::float32) {
      throw std::runtime_error("expert_selector: router_output must be float32");
    }

    const float* data = router_output.data_ptr<float>();
    int64_t batch = router_output.dim(0);
    int64_t seq_len = router_output.dim(1);
    int64_t top_k = router_output.dim(2);

    // Collect unique expert IDs using unordered_set for O(1) insertion
    std::unordered_set<int> selected_experts;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t k = 0; k < top_k; ++k) {
          int64_t idx = ((b * seq_len + s) * top_k + k) * 2;
          int expert_id = static_cast<int>(data[idx]);
          selected_experts.insert(expert_id);
        }
      }
    }

    // Convert to int32 tensor [num_selected]
    int64_t num_selected = static_cast<int64_t>(selected_experts.size());
    dynamic_tensor result(dtype::int32, {num_selected});
    int32_t* result_data = result.data_ptr<int32_t>();
    
    int idx = 0;
    for (int expert_id : selected_experts) {
      result_data[idx++] = expert_id;
    }

    return result;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_selector_node, coalsack::graph_node)
