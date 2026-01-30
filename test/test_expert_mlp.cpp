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

// GELU function for reference
float gelu_ref(float x) {
  constexpr float sqrt_2_over_pi = 0.7978845608028654f;
  constexpr float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// Test basic expert MLP operation
bool test_expert_mlp_basic() {
  std::cout << "Test 1: Basic expert MLP\n";

  // Simple setup: batch=1, seq_len=2, hidden_dim=8, expert_ffn_dim=16
  int64_t batch = 1;
  int64_t seq_len = 2;
  int64_t hidden_dim = 8;
  int64_t expert_ffn_dim = 16;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w1_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> w2_shape = {expert_ffn_dim, hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w1(dtype::float32, w1_shape);
  dynamic_tensor w2(dtype::float32, w2_shape);

  fill_random(hidden_states, -0.5f, 0.5f);
  fill_random(w1, -0.1f, 0.1f);
  fill_random(w2, -0.1f, 0.1f);

  // Create expert MLP node
  expert_mlp_node node(0);

  // Compute
  std::vector<dynamic_tensor> inputs = {hidden_states, w1, w2};
  dynamic_tensor output = node.compute_test(inputs);

  // Verify output shape matches input shape
  if (output.shape() != hidden_states.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify output is not all zeros
  const float* output_data = output.data_ptr<float>();
  bool has_nonzero = false;
  for (int64_t i = 0; i < output.numel(); ++i) {
    if (std::abs(output_data[i]) > 1e-6f) {
      has_nonzero = true;
      break;
    }
  }

  if (!has_nonzero) {
    std::cerr << "  ERROR: Output is all zeros\n";
    return false;
  }

  std::cout << "  ✓ Basic expert MLP works\n";
  return true;
}

// Test GELU activation
bool test_gelu_activation() {
  std::cout << "\nTest 2: GELU activation\n";

  // Create simple case where we can verify GELU is applied
  int64_t batch = 1;
  int64_t seq_len = 1;
  int64_t hidden_dim = 2;
  int64_t expert_ffn_dim = 2;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w1_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> w2_shape = {expert_ffn_dim, hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w1(dtype::float32, w1_shape);
  dynamic_tensor w2(dtype::float32, w2_shape);

  // Set specific values
  float* hidden_data = hidden_states.data_ptr<float>();
  hidden_data[0] = 1.0f;
  hidden_data[1] = 0.0f;

  // Set w1 to identity-like (with some scaling)
  float* w1_data = w1.data_ptr<float>();
  for (int64_t i = 0; i < w1.numel(); ++i) w1_data[i] = 0.0f;
  w1_data[0] = 1.0f;  // [0, 0] -> 1.0
  w1_data[3] = 1.0f;  // [1, 1] -> 1.0

  // Set w2 to identity
  float* w2_data = w2.data_ptr<float>();
  for (int64_t i = 0; i < w2.numel(); ++i) w2_data[i] = 0.0f;
  w2_data[0] = 1.0f;  // [0, 0] -> 1.0
  w2_data[3] = 1.0f;  // [1, 1] -> 1.0

  // Create expert MLP node
  expert_mlp_node node(0);

  // Compute
  std::vector<dynamic_tensor> inputs = {hidden_states, w1, w2};
  dynamic_tensor output = node.compute_test(inputs);

  const float* output_data = output.data_ptr<float>();

  // Expected: output[0] = gelu(1.0), output[1] = gelu(0.0)
  float expected_0 = gelu_ref(1.0f);
  float expected_1 = gelu_ref(0.0f);

  float diff_0 = std::abs(output_data[0] - expected_0);
  float diff_1 = std::abs(output_data[1] - expected_1);

  if (diff_0 > 1e-5f) {
    std::cerr << "  ERROR: GELU not applied correctly at position 0: expected=" << expected_0
              << ", got=" << output_data[0] << ", diff=" << diff_0 << "\n";
    return false;
  }

  if (diff_1 > 1e-5f) {
    std::cerr << "  ERROR: GELU not applied correctly at position 1: expected=" << expected_1
              << ", got=" << output_data[1] << ", diff=" << diff_1 << "\n";
    return false;
  }

  std::cout << "  ✓ GELU activation correct\n";
  return true;
}

// Test multiple experts
bool test_multiple_experts() {
  std::cout << "\nTest 3: Multiple expert IDs\n";

  int64_t batch = 1;
  int64_t seq_len = 2;
  int64_t hidden_dim = 8;
  int64_t expert_ffn_dim = 16;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w1_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> w2_shape = {expert_ffn_dim, hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w1(dtype::float32, w1_shape);
  dynamic_tensor w2(dtype::float32, w2_shape);

  fill_random(hidden_states);
  fill_random(w1);
  fill_random(w2);

  // Create multiple expert nodes with different IDs
  std::vector<int> expert_ids = {0, 5, 15, 31};

  for (int expert_id : expert_ids) {
    expert_mlp_node node(expert_id);

    if (node.get_expert_id() != expert_id) {
      std::cerr << "  ERROR: Expert ID mismatch: expected=" << expert_id
                << ", got=" << node.get_expert_id() << "\n";
      return false;
    }

    // Compute
    std::vector<dynamic_tensor> inputs = {hidden_states, w1, w2};
    dynamic_tensor output = node.compute_test(inputs);

    // Verify output shape
    if (output.shape() != hidden_states.shape()) {
      std::cerr << "  ERROR: Output shape mismatch for expert " << expert_id << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Multiple experts work correctly\n";
  return true;
}

// Test GPT-OSS scale dimensions
bool test_gpt_oss_scale() {
  std::cout << "\nTest 4: GPT-OSS scale dimensions\n";

  // GPT-OSS: hidden_dim=2880, expert_ffn_dim=2880
  int64_t batch = 2;
  int64_t seq_len = 4;
  int64_t hidden_dim = 128;     // Using smaller for testing
  int64_t expert_ffn_dim = 128;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w1_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> w2_shape = {expert_ffn_dim, hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w1(dtype::float32, w1_shape);
  dynamic_tensor w2(dtype::float32, w2_shape);

  fill_random(hidden_states);
  fill_random(w1, -0.05f, 0.05f);
  fill_random(w2, -0.05f, 0.05f);

  // Create expert MLP node
  expert_mlp_node node(0);

  // Compute
  std::vector<dynamic_tensor> inputs = {hidden_states, w1, w2};
  dynamic_tensor output = node.compute_test(inputs);

  // Verify output shape
  if (output.shape() != hidden_states.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify output is not NaN or all zeros
  const float* output_data = output.data_ptr<float>();
  bool has_nan = false;
  bool has_nonzero = false;

  for (int64_t i = 0; i < output.numel(); ++i) {
    if (std::isnan(output_data[i])) {
      has_nan = true;
      break;
    }
    if (std::abs(output_data[i]) > 1e-6f) {
      has_nonzero = true;
    }
  }

  if (has_nan) {
    std::cerr << "  ERROR: Output contains NaN values\n";
    return false;
  }

  if (!has_nonzero) {
    std::cerr << "  ERROR: Output is all zeros\n";
    return false;
  }

  std::cout << "  ✓ GPT-OSS scale works correctly\n";
  return true;
}

int main() {
  std::cout << "Testing Expert MLP Node\n";
  std::cout << "=======================\n\n";

  bool test1 = test_expert_mlp_basic();
  bool test2 = test_gelu_activation();
  bool test3 = test_multiple_experts();
  bool test4 = test_gpt_oss_scale();

  if (test1 && test2 && test3 && test4) {
    std::cout << "\n✓ All tests passed!\n";
    return 0;
  } else {
    std::cerr << "\n✗ Some tests failed\n";
    return 1;
  }
}
