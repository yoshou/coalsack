#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "coalsack/nn/nn_nodes.h"

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

// Test basic RMSNorm operation
bool test_rmsnorm_basic() {
  std::cout << "Test 1: Basic RMSNorm operation\n";

  // Create input tensor [2, 4, 8]
  std::vector<int64_t> input_shape = {2, 4, 8};
  dynamic_tensor input(dtype::float32, input_shape);
  fill_random(input);

  // Create weight tensor [8]
  std::vector<int64_t> weight_shape = {8};
  dynamic_tensor weight(dtype::float32, weight_shape);
  float* weight_data = weight.data_ptr<float>();
  for (int64_t i = 0; i < 8; ++i) {
    weight_data[i] = 1.0f;  // Initialize to 1.0
  }

  // Create RMSNorm node
  rmsnorm_node node;
  node.set_epsilon(1e-5f);

  // Compute
  dynamic_tensor output = node.compute_test(input, weight);

  // Verify output shape
  if (output.shape() != input.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify RMS normalization
  const float* input_data = input.data_ptr<float>();
  const float* output_data = output.data_ptr<float>();

  bool all_passed = true;
  int64_t hidden_dim = 8;
  int64_t outer_size = 2 * 4;

  for (int64_t i = 0; i < outer_size; ++i) {
    // Compute expected RMS
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < hidden_dim; ++j) {
      float val = input_data[i * hidden_dim + j];
      sum_sq += val * val;
    }
    float rms = std::sqrt(sum_sq / hidden_dim + 1e-5f);

    // Verify each element
    for (int64_t j = 0; j < hidden_dim; ++j) {
      float expected = input_data[i * hidden_dim + j] / rms;
      float actual = output_data[i * hidden_dim + j];
      float diff = std::abs(expected - actual);

      if (diff > 1e-5f) {
        std::cerr << "  ERROR: Mismatch at [" << i << ", " << j << "]: expected=" << expected
                  << ", actual=" << actual << ", diff=" << diff << "\n";
        all_passed = false;
      }
    }
  }

  if (all_passed) {
    std::cout << "  ✓ RMS normalization correct\n";
  }

  return all_passed;
}

// Test RMSNorm with weight scaling
bool test_rmsnorm_with_weight() {
  std::cout << "\nTest 2: RMSNorm with weight scaling\n";

  // Create input tensor [1, 4]
  std::vector<int64_t> input_shape = {1, 4};
  dynamic_tensor input(dtype::float32, input_shape);
  float* input_data = input.data_ptr<float>();
  input_data[0] = 1.0f;
  input_data[1] = 2.0f;
  input_data[2] = 3.0f;
  input_data[3] = 4.0f;

  // Create weight tensor [4]
  std::vector<int64_t> weight_shape = {4};
  dynamic_tensor weight(dtype::float32, weight_shape);
  float* weight_data = weight.data_ptr<float>();
  weight_data[0] = 2.0f;
  weight_data[1] = 0.5f;
  weight_data[2] = 1.5f;
  weight_data[3] = 1.0f;

  // Create RMSNorm node
  rmsnorm_node node;
  node.set_epsilon(0.0f);  // No epsilon for this test

  // Compute
  dynamic_tensor output = node.compute_test(input, weight);

  // Manual calculation
  // sum_sq = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
  // rms = sqrt(30 / 4) = sqrt(7.5) = 2.73861...
  // normalized = [1/2.73861, 2/2.73861, 3/2.73861, 4/2.73861]
  // output = normalized * weight

  float expected_rms = std::sqrt(30.0f / 4.0f);
  const float* output_data = output.data_ptr<float>();

  std::vector<float> expected = {1.0f / expected_rms * 2.0f, 2.0f / expected_rms * 0.5f,
                                 3.0f / expected_rms * 1.5f, 4.0f / expected_rms * 1.0f};

  bool all_passed = true;
  for (int64_t i = 0; i < 4; ++i) {
    float diff = std::abs(expected[i] - output_data[i]);
    if (diff > 1e-5f) {
      std::cerr << "  ERROR: Mismatch at [" << i << "]: expected=" << expected[i]
                << ", actual=" << output_data[i] << ", diff=" << diff << "\n";
      all_passed = false;
    }
  }

  if (all_passed) {
    std::cout << "  ✓ Weight scaling correct\n";
  }

  return all_passed;
}

// Test edge case: single element
bool test_rmsnorm_single_element() {
  std::cout << "\nTest 3: Single element normalization\n";

  // Create input tensor [1, 1]
  std::vector<int64_t> input_shape = {1, 1};
  dynamic_tensor input(dtype::float32, input_shape);
  float* input_data = input.data_ptr<float>();
  input_data[0] = 5.0f;

  // Create weight tensor [1]
  std::vector<int64_t> weight_shape = {1};
  dynamic_tensor weight(dtype::float32, weight_shape);
  float* weight_data = weight.data_ptr<float>();
  weight_data[0] = 2.0f;

  // Create RMSNorm node
  rmsnorm_node node;
  node.set_epsilon(0.0f);

  // Compute
  dynamic_tensor output = node.compute_test(input, weight);

  // Expected: 5 / sqrt(25) * 2 = 5 / 5 * 2 = 2
  const float* output_data = output.data_ptr<float>();
  float expected = 2.0f;
  float diff = std::abs(expected - output_data[0]);

  if (diff > 1e-5f) {
    std::cerr << "  ERROR: Expected " << expected << ", got " << output_data[0] << "\n";
    return false;
  }

  std::cout << "  ✓ Single element correct\n";
  return true;
}

int main() {
  std::cout << "Testing RMSNorm Node\n";
  std::cout << "====================\n\n";

  bool test1 = test_rmsnorm_basic();
  bool test2 = test_rmsnorm_with_weight();
  bool test3 = test_rmsnorm_single_element();

  if (test1 && test2 && test3) {
    std::cout << "\n✓ All tests passed!\n";
    return 0;
  } else {
    std::cerr << "\n✗ Some tests failed\n";
    return 1;
  }
}
