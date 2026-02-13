#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "nn_nodes.h"

using namespace coalsack;

// Helper function to fill tensor with random values
void fill_random(dynamic_tensor& tensor, float min_val = -1.0f, float max_val = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dis(min_val, max_val);

  float* data = tensor.data_ptr<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    data[i] = dis(gen);
  }
}

// Test basic MoE routing
bool test_moe_router_basic() {
  std::cout << "Test 1: Basic MoE routing\n";

  // Simple setup: batch=1, seq_len=2, hidden_dim=4, 8 experts, top-2
  int64_t batch = 1;
  int64_t seq_len = 2;
  int64_t hidden_dim = 4;
  int64_t num_experts = 8;
  int64_t top_k = 2;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> gate_shape = {num_experts, hidden_dim};
  std::vector<int64_t> bias_shape = {num_experts};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor gate_weights(dtype::float32, gate_shape);
  dynamic_tensor gate_bias(dtype::float32, bias_shape);

  fill_random(hidden_states);
  fill_random(gate_weights);
  {
    float* b = gate_bias.data_ptr<float>();
    for (int64_t i = 0; i < gate_bias.numel(); ++i) {
      b[i] = 0.0f;
    }
  }

  // Create MoE router node
  moe_router_node node;
  node.set_config(num_experts, top_k);

  // Compute
  dynamic_tensor output = node.compute_test(hidden_states, gate_weights, gate_bias);

  // Verify output shape: [batch, seq_len, top_k, 2]
  std::vector<int64_t> expected_shape = {batch, seq_len, top_k, 2};
  if (output.shape() != expected_shape) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  const float* out_data = output.data_ptr<float>();

  // Verify expert indices are valid and weights sum to ~1
  for (int64_t s = 0; s < seq_len; ++s) {
    float weight_sum = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
      int64_t idx = (s * top_k + k) * 2;
      float expert_index = out_data[idx + 0];
      float weight = out_data[idx + 1];

      // Check expert index is valid
      if (expert_index < 0 || expert_index >= num_experts) {
        std::cerr << "  ERROR: Invalid expert index: " << expert_index << "\n";
        return false;
      }

      // Check weight is positive
      if (weight <= 0.0f) {
        std::cerr << "  ERROR: Non-positive weight: " << weight << "\n";
        return false;
      }

      weight_sum += weight;
    }

    // Check weights sum to ~1
    if (std::abs(weight_sum - 1.0f) > 1e-5f) {
      std::cerr << "  ERROR: Weights don't sum to 1: " << weight_sum << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Basic MoE routing works\n";
  return true;
}

// Test top-k selection correctness
bool test_top_k_selection() {
  std::cout << "\nTest 2: Top-k selection correctness\n";

  // Create a controlled scenario where we know which experts should be selected
  int64_t batch = 1;
  int64_t seq_len = 1;
  int64_t hidden_dim = 4;
  int64_t num_experts = 6;
  int64_t top_k = 3;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> gate_shape = {num_experts, hidden_dim};
  std::vector<int64_t> bias_shape = {num_experts};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor gate_weights(dtype::float32, gate_shape);
  dynamic_tensor gate_bias(dtype::float32, bias_shape);

  // Set hidden_states to all 1s
  float* hidden_data = hidden_states.data_ptr<float>();
  for (int64_t i = 0; i < hidden_states.numel(); ++i) {
    hidden_data[i] = 1.0f;
  }

  // Set gate_weights so that each expert's score is its index
  // (expert 5 gets highest score, expert 0 gets lowest)
  float* gate_data = gate_weights.data_ptr<float>();
  for (int64_t e = 0; e < num_experts; ++e) {
    for (int64_t d = 0; d < hidden_dim; ++d) {
      gate_data[e * hidden_dim + d] = static_cast<float>(e) / hidden_dim;
    }
  }

  // Zero bias
  {
    float* b = gate_bias.data_ptr<float>();
    for (int64_t i = 0; i < gate_bias.numel(); ++i) {
      b[i] = 0.0f;
    }
  }

  // Create MoE router node
  moe_router_node node;
  node.set_config(num_experts, top_k);

  // Compute
  dynamic_tensor output = node.compute_test(hidden_states, gate_weights, gate_bias);

  const float* out_data = output.data_ptr<float>();

  // Expected top-3 experts: 5, 4, 3 (in that order)
  std::vector<int> expected_experts = {5, 4, 3};

  for (int64_t k = 0; k < top_k; ++k) {
    int64_t idx = k * 2;
    int expert_index = static_cast<int>(out_data[idx + 0]);

    if (expert_index != expected_experts[k]) {
      std::cerr << "  ERROR: Expected expert " << expected_experts[k] << " at position " << k
                << ", got " << expert_index << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Top-k selection correct\n";
  return true;
}

// Test GPT-OSS scale (32 experts, top-4)
bool test_gpt_oss_scale() {
  std::cout << "\nTest 3: GPT-OSS scale (32 experts, top-4)\n";

  int64_t batch = 2;
  int64_t seq_len = 8;
  int64_t hidden_dim = 128;
  int64_t num_experts = 32;
  int64_t top_k = 4;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> gate_shape = {num_experts, hidden_dim};
  std::vector<int64_t> bias_shape = {num_experts};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor gate_weights(dtype::float32, gate_shape);
  dynamic_tensor gate_bias(dtype::float32, bias_shape);

  fill_random(hidden_states);
  fill_random(gate_weights);
  {
    float* b = gate_bias.data_ptr<float>();
    for (int64_t i = 0; i < gate_bias.numel(); ++i) {
      b[i] = 0.0f;
    }
  }

  // Create MoE router node
  moe_router_node node;
  node.set_config(num_experts, top_k);

  // Compute
  dynamic_tensor output = node.compute_test(hidden_states, gate_weights, gate_bias);

  // Verify output shape
  std::vector<int64_t> expected_shape = {batch, seq_len, top_k, 2};
  if (output.shape() != expected_shape) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  const float* out_data = output.data_ptr<float>();

  // Verify all positions have valid indices and normalized weights
  bool all_valid = true;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      float weight_sum = 0.0f;
      int64_t base_idx = ((b * seq_len + s) * top_k) * 2;

      for (int64_t k = 0; k < top_k; ++k) {
        int64_t idx = base_idx + k * 2;
        int expert_index = static_cast<int>(out_data[idx + 0]);
        float weight = out_data[idx + 1];

        if (expert_index < 0 || expert_index >= num_experts) {
          std::cerr << "  ERROR: Invalid expert index at [" << b << ", " << s << ", " << k
                    << "]: " << expert_index << "\n";
          all_valid = false;
        }

        if (weight <= 0.0f || weight > 1.0f) {
          std::cerr << "  ERROR: Invalid weight at [" << b << ", " << s << ", " << k
                    << "]: " << weight << "\n";
          all_valid = false;
        }

        weight_sum += weight;
      }

      if (std::abs(weight_sum - 1.0f) > 1e-4f) {
        std::cerr << "  ERROR: Weights don't sum to 1 at [" << b << ", " << s << "]: " << weight_sum
                  << "\n";
        all_valid = false;
      }
    }
  }

  if (!all_valid) {
    return false;
  }

  std::cout << "  ✓ GPT-OSS scale works correctly\n";
  return true;
}

// Test weight normalization
bool test_weight_normalization() {
  std::cout << "\nTest 4: Weight normalization\n";

  int64_t batch = 1;
  int64_t seq_len = 3;
  int64_t hidden_dim = 8;
  int64_t num_experts = 10;
  int64_t top_k = 5;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> gate_shape = {num_experts, hidden_dim};
  std::vector<int64_t> bias_shape = {num_experts};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor gate_weights(dtype::float32, gate_shape);
  dynamic_tensor gate_bias(dtype::float32, bias_shape);

  fill_random(hidden_states);
  fill_random(gate_weights);
  {
    float* b = gate_bias.data_ptr<float>();
    for (int64_t i = 0; i < gate_bias.numel(); ++i) {
      b[i] = 0.0f;
    }
  }

  // Create MoE router node
  moe_router_node node;
  node.set_config(num_experts, top_k);

  // Compute
  dynamic_tensor output = node.compute_test(hidden_states, gate_weights, gate_bias);

  const float* out_data = output.data_ptr<float>();

  // Verify weights are normalized for each token
  for (int64_t s = 0; s < seq_len; ++s) {
    float weight_sum = 0.0f;
    int64_t base_idx = (s * top_k) * 2;

    for (int64_t k = 0; k < top_k; ++k) {
      float weight = out_data[base_idx + k * 2 + 1];
      weight_sum += weight;
    }

    if (std::abs(weight_sum - 1.0f) > 1e-5f) {
      std::cerr << "  ERROR: Weights not normalized at position " << s << ": sum=" << weight_sum
                << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Weight normalization correct\n";
  return true;
}

int main() {
  std::cout << "Testing MoE Router Node\n";
  std::cout << "=======================\n\n";

  bool test1 = test_moe_router_basic();
  bool test2 = test_top_k_selection();
  bool test3 = test_gpt_oss_scale();
  bool test4 = test_weight_normalization();

  if (test1 && test2 && test3 && test4) {
    std::cout << "\n✓ All tests passed!\n";
    return 0;
  } else {
    std::cerr << "\n✗ Some tests failed\n";
    return 1;
  }
}
