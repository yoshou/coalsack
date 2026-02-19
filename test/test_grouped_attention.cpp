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

// Test basic grouped attention
bool test_grouped_attention_basic() {
  std::cout << "Test 1: Basic grouped attention\n";

  // Create simple setup: batch=1, seq_len=2, 4 Q heads, 2 KV heads, head_dim=4
  int64_t batch = 1;
  int64_t seq_len = 2;
  int64_t num_q_heads = 4;
  int64_t num_kv_heads = 2;
  int64_t head_dim = 4;

  std::vector<int64_t> q_shape = {batch, seq_len, num_q_heads * head_dim};
  std::vector<int64_t> kv_shape = {batch, seq_len, num_kv_heads * head_dim};

  dynamic_tensor query(dtype::float32, q_shape);
  dynamic_tensor key(dtype::float32, kv_shape);
  dynamic_tensor value(dtype::float32, kv_shape);

  fill_random(query, -0.5f, 0.5f);
  fill_random(key, -0.5f, 0.5f);
  fill_random(value, 0.0f, 1.0f);

  // Create grouped attention node
  grouped_attention_node node;
  node.set_config(num_q_heads, num_kv_heads, head_dim);

  // Compute
  std::vector<dynamic_tensor> inputs = {query, key, value};
  dynamic_tensor output = node.compute_test(inputs);

  // Verify output shape matches query shape
  if (output.shape() != query.shape()) {
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

  std::cout << "  ✓ Basic grouped attention works\n";
  return true;
}

// Test causal masking
bool test_causal_masking() {
  std::cout << "\nTest 2: Causal masking\n";

  // Create simple case where we can verify causality
  int64_t batch = 1;
  int64_t seq_len = 3;
  int64_t num_q_heads = 2;
  int64_t num_kv_heads = 1;
  int64_t head_dim = 2;

  std::vector<int64_t> q_shape = {batch, seq_len, num_q_heads * head_dim};
  std::vector<int64_t> kv_shape = {batch, seq_len, num_kv_heads * head_dim};

  dynamic_tensor query(dtype::float32, q_shape);
  dynamic_tensor key(dtype::float32, kv_shape);
  dynamic_tensor value(dtype::float32, kv_shape);

  // Set specific values to test causality
  float* q_data = query.data_ptr<float>();
  float* k_data = key.data_ptr<float>();
  float* v_data = value.data_ptr<float>();

  // Initialize with known values
  for (int64_t i = 0; i < query.numel(); ++i) q_data[i] = 0.1f;
  for (int64_t i = 0; i < key.numel(); ++i) k_data[i] = 0.1f;

  // Set value to position index so we can detect if future info leaks
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      v_data[pos * head_dim + d] = static_cast<float>(pos + 1);  // 1, 2, 3
    }
  }

  // Create grouped attention node
  grouped_attention_node node;
  node.set_config(num_q_heads, num_kv_heads, head_dim);

  // Compute
  std::vector<dynamic_tensor> inputs = {query, key, value};
  dynamic_tensor output = node.compute_test(inputs);

  const float* output_data = output.data_ptr<float>();

  // At position 0, output should only depend on value[0] (which is all 1s)
  // At position 1, output should depend on value[0] and value[1]
  // At position 2, output should depend on all values

  // Since all Q and K are the same (0.1), attention weights should be uniform
  // Position 0: should be close to 1.0 (only attends to value[0])
  // Position 1: should be close to (1+2)/2 = 1.5 (attends to value[0] and value[1])
  // Position 2: should be close to (1+2+3)/3 = 2.0 (attends to all)

  float pos0_avg = 0.0f;
  float pos1_avg = 0.0f;
  float pos2_avg = 0.0f;

  for (int64_t h = 0; h < num_q_heads; ++h) {
    for (int64_t d = 0; d < head_dim; ++d) {
      pos0_avg += output_data[0 * num_q_heads * head_dim + h * head_dim + d];
      pos1_avg += output_data[1 * num_q_heads * head_dim + h * head_dim + d];
      pos2_avg += output_data[2 * num_q_heads * head_dim + h * head_dim + d];
    }
  }

  pos0_avg /= (num_q_heads * head_dim);
  pos1_avg /= (num_q_heads * head_dim);
  pos2_avg /= (num_q_heads * head_dim);

  // Check ordering: pos0_avg < pos1_avg < pos2_avg (due to causality)
  if (pos0_avg >= pos1_avg || pos1_avg >= pos2_avg) {
    std::cerr << "  ERROR: Causal ordering violated: pos0=" << pos0_avg << ", pos1=" << pos1_avg
              << ", pos2=" << pos2_avg << "\n";
    return false;
  }

  std::cout << "  ✓ Causal masking works correctly\n";
  return true;
}

// Test grouped heads (multiple Q heads per KV head)
bool test_grouped_heads() {
  std::cout << "\nTest 3: Grouped heads (8 Q heads, 2 KV heads)\n";

  int64_t batch = 2;
  int64_t seq_len = 4;
  int64_t num_q_heads = 8;
  int64_t num_kv_heads = 2;
  int64_t head_dim = 8;

  std::vector<int64_t> q_shape = {batch, seq_len, num_q_heads * head_dim};
  std::vector<int64_t> kv_shape = {batch, seq_len, num_kv_heads * head_dim};

  dynamic_tensor query(dtype::float32, q_shape);
  dynamic_tensor key(dtype::float32, kv_shape);
  dynamic_tensor value(dtype::float32, kv_shape);

  fill_random(query);
  fill_random(key);
  fill_random(value);

  // Create grouped attention node
  grouped_attention_node node;
  node.set_config(num_q_heads, num_kv_heads, head_dim);

  // Compute
  std::vector<dynamic_tensor> inputs = {query, key, value};
  dynamic_tensor output = node.compute_test(inputs);

  // Verify output shape
  if (output.shape() != query.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  std::cout << "  ✓ Grouped heads handled correctly\n";
  return true;
}

// Test GPT-OSS scale (64 Q heads, 8 KV heads)
bool test_gpt_oss_scale() {
  std::cout << "\nTest 4: GPT-OSS scale (64 Q heads, 8 KV heads)\n";

  int64_t batch = 1;
  int64_t seq_len = 8;
  int64_t num_q_heads = 64;
  int64_t num_kv_heads = 8;
  int64_t head_dim = 45;  // 2880 / 64

  std::vector<int64_t> q_shape = {batch, seq_len, num_q_heads * head_dim};
  std::vector<int64_t> kv_shape = {batch, seq_len, num_kv_heads * head_dim};

  dynamic_tensor query(dtype::float32, q_shape);
  dynamic_tensor key(dtype::float32, kv_shape);
  dynamic_tensor value(dtype::float32, kv_shape);

  fill_random(query);
  fill_random(key);
  fill_random(value);

  // Create grouped attention node
  grouped_attention_node node;
  node.set_config(num_q_heads, num_kv_heads, head_dim);

  // Compute
  std::vector<dynamic_tensor> inputs = {query, key, value};
  dynamic_tensor output = node.compute_test(inputs);

  // Verify output shape
  if (output.shape() != query.shape()) {
    std::cerr << "  ERROR: Output shape mismatch\n";
    return false;
  }

  // Verify output is not all zeros or NaNs
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
  std::cout << "Testing Grouped Attention Node\n";
  std::cout << "===============================\n\n";

  bool test1 = test_grouped_attention_basic();
  bool test2 = test_causal_masking();
  bool test3 = test_grouped_heads();
  bool test4 = test_gpt_oss_scale();

  if (test1 && test2 && test3 && test4) {
    std::cout << "\n✓ All tests passed!\n";
    return 0;
  } else {
    std::cerr << "\n✗ Some tests failed\n";
    return 1;
  }
}
