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

// Helper to compute L2 norm of a vector
float compute_norm(const float* data, int64_t size) {
  float sum = 0.0f;
  for (int64_t i = 0; i < size; ++i) {
    sum += data[i] * data[i];
  }
  return std::sqrt(sum);
}

// Test basic RoPE operation
bool test_rope_basic() {
  std::cout << "Test 1: Basic RoPE operation\n";

  // Create input tensor [1, 1, 4, 8] (batch=1, heads=1, seq_len=4, head_dim=8)
  std::vector<int64_t> input_shape = {1, 1, 4, 8};
  dynamic_tensor input(dtype::float32, input_shape);
  fill_random(input);

  // Create RoPE node
  rope_node node;
  node.set_config(8, 128);  // head_dim=8, max_seq_len=128

  // Compute
  dynamic_tensor output = node.compute_test(input);

  // Verify output shape
  if (output.shape() != input.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify that rotation preserves norms (rotation doesn't change magnitude)
  const float* input_data = input.data_ptr<float>();
  const float* output_data = output.data_ptr<float>();

  bool all_passed = true;
  int64_t head_dim = 8;
  int64_t seq_len = 4;

  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float input_norm = compute_norm(input_data + pos * head_dim, head_dim);
    float output_norm = compute_norm(output_data + pos * head_dim, head_dim);
    float diff = std::abs(input_norm - output_norm);

    if (diff > 1e-5f) {
      std::cerr << "  ERROR: Norm not preserved at pos " << pos << ": input_norm=" << input_norm
                << ", output_norm=" << output_norm << ", diff=" << diff << "\n";
      all_passed = false;
    }
  }

  if (all_passed) {
    std::cout << "  ✓ RoPE preserves norms correctly\n";
  }

  return all_passed;
}

// Test RoPE rotation correctness with known values
bool test_rope_rotation() {
  std::cout << "\nTest 2: RoPE rotation correctness\n";

  // Create simple input [1, 1, 2, 4] with known values
  std::vector<int64_t> input_shape = {1, 1, 2, 4};
  dynamic_tensor input(dtype::float32, input_shape);
  float* input_data = input.data_ptr<float>();

  // Position 0: [1, 0, 1, 0]
  // NeoX-style Split: pairs are (x[0], x[2]) and (x[1], x[3])
  input_data[0] = 1.0f;  // first half, dim 0
  input_data[1] = 0.0f;  // first half, dim 1
  input_data[2] = 1.0f;  // second half, dim 0
  input_data[3] = 0.0f;  // second half, dim 1

  // Position 1: [1, 0, 1, 0]
  input_data[4] = 1.0f;
  input_data[5] = 0.0f;
  input_data[6] = 1.0f;
  input_data[7] = 0.0f;

  // Create RoPE node with base=10000
  rope_node node;
  node.set_config(4, 128, 10000.0f, 1.0f);  // head_dim=4, no scaling

  // Compute
  dynamic_tensor output = node.compute_test(input);

  const float* output_data = output.data_ptr<float>();

  // At position 0, angle for dimension 0 is 0, so rotation by 0 should keep [1,1]
  // freq_0 = 10000^(-2*0/4) = 10000^0 = 1
  // angle_0 = 0 * 1 = 0
  // cos(0) = 1, sin(0) = 0
  // Pair (x[0]=1, x[2]=1) rotated by angle 0
  // → out[0] = 1*1 - 1*0 = 1, out[2] = 1*0 + 1*1 = 1

  // Check position 0, first half dim 0
  float expected_00 = 1.0f;  // cos(0) * 1 - sin(0) * 1 = 1
  float expected_02 = 1.0f;  // sin(0) * 1 + cos(0) * 1 = 1
  if (std::abs(output_data[0] - expected_00) > 1e-5f ||
      std::abs(output_data[2] - expected_02) > 1e-5f) {
    std::cerr << "  ERROR: Position 0, pair 0 mismatch: expected [" << expected_00 << ", "
              << expected_02 << "], got [" << output_data[0] << ", " << output_data[2] << "]\n";
    return false;
  }

  // At position 1, angle for dimension 0 is 1 * freq_0 = 1
  // Pair (x[4]=1, x[6]=1) rotated by angle 1
  // → out[4] = cos(1)*1 - sin(1)*1, out[6] = sin(1)*1 + cos(1)*1
  float angle1 = 1.0f;
  float expected_10 = std::cos(angle1) - std::sin(angle1);  // ~-0.301
  float expected_12 = std::sin(angle1) + std::cos(angle1);  // ~1.382

  if (std::abs(output_data[4] - expected_10) > 1e-5f ||
      std::abs(output_data[6] - expected_12) > 1e-5f) {
    std::cerr << "  ERROR: Position 1, pair 0 mismatch: expected [" << expected_10 << ", "
              << expected_12 << "], got [" << output_data[4] << ", " << output_data[6] << "]\n";
    return false;
  }

  std::cout << "  ✓ Rotation angles correct\n";
  return true;
}

// Test YaRN scaling
bool test_rope_yarn_scaling() {
  std::cout << "\nTest 3: RoPE with YaRN scaling\n";

  // Create input tensor [1, 1, 2, 8]
  std::vector<int64_t> input_shape = {1, 1, 2, 8};
  dynamic_tensor input(dtype::float32, input_shape);
  fill_random(input);

  // Create RoPE node with YaRN scaling (scaling_factor > 1)
  rope_node node;
  node.set_config(8, 128, 150000.0f, 32.0f);  // GPT-OSS config

  // Compute
  dynamic_tensor output = node.compute_test(input);

  // Verify output shape
  if (output.shape() != input.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify norms are preserved
  const float* input_data = input.data_ptr<float>();
  const float* output_data = output.data_ptr<float>();

  bool all_passed = true;
  int64_t head_dim = 8;
  int64_t seq_len = 2;

  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float input_norm = compute_norm(input_data + pos * head_dim, head_dim);
    float output_norm = compute_norm(output_data + pos * head_dim, head_dim);
    float diff = std::abs(input_norm - output_norm);

    if (diff > 1e-5f) {
      std::cerr << "  ERROR: Norm not preserved at pos " << pos << ": diff=" << diff << "\n";
      all_passed = false;
    }
  }

  if (all_passed) {
    std::cout << "  ✓ YaRN scaling works correctly\n";
  }

  return all_passed;
}

// Test multiple heads
bool test_rope_multiple_heads() {
  std::cout << "\nTest 4: RoPE with multiple heads\n";

  // Create input tensor [2, 4, 3, 16] (batch=2, heads=4, seq_len=3, head_dim=16)
  std::vector<int64_t> input_shape = {2, 4, 3, 16};
  dynamic_tensor input(dtype::float32, input_shape);
  fill_random(input);

  // Create RoPE node
  rope_node node;
  node.set_config(16, 128);

  // Compute
  dynamic_tensor output = node.compute_test(input);

  // Verify output shape
  if (output.shape() != input.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify norms are preserved for all heads and batches
  const float* input_data = input.data_ptr<float>();
  const float* output_data = output.data_ptr<float>();

  bool all_passed = true;
  int64_t batch = 2;
  int64_t num_heads = 4;
  int64_t seq_len = 3;
  int64_t head_dim = 16;

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < num_heads; ++h) {
      for (int64_t pos = 0; pos < seq_len; ++pos) {
        int64_t idx = ((b * num_heads + h) * seq_len + pos) * head_dim;
        float input_norm = compute_norm(input_data + idx, head_dim);
        float output_norm = compute_norm(output_data + idx, head_dim);
        float diff = std::abs(input_norm - output_norm);

        if (diff > 1e-4f) {
          std::cerr << "  ERROR: Norm not preserved at batch=" << b << ", head=" << h
                    << ", pos=" << pos << ": diff=" << diff << "\n";
          all_passed = false;
        }
      }
    }
  }

  if (all_passed) {
    std::cout << "  ✓ Multiple heads handled correctly\n";
  }

  return all_passed;
}

int main() {
  std::cout << "Testing RoPE Node\n";
  std::cout << "=================\n\n";

  bool test1 = test_rope_basic();
  bool test2 = test_rope_rotation();
  bool test3 = test_rope_yarn_scaling();
  bool test4 = test_rope_multiple_heads();

  if (test1 && test2 && test3 && test4) {
    std::cout << "\n✓ All tests passed!\n";
    return 0;
  } else {
    std::cerr << "\n✗ Some tests failed\n";
    return 1;
  }
}
