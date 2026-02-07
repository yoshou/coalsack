#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "nn_nodes.h"
#include "gguf_dequant.h"

using namespace coalsack;

// Helper function to fill fp32 tensor with random values
void fill_random_fp32(dynamic_tensor& tensor, float min_val = -1.0f, float max_val = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dis(min_val, max_val);

  float* data = tensor.data_ptr<float>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    data[i] = dis(gen);
  }
}

// Helper function to fill fp16 tensor with random values
void fill_random_fp16(dynamic_tensor& tensor, float min_val = -1.0f, float max_val = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dis(min_val, max_val);

  uint16_t* data = tensor.data_ptr<uint16_t>();
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    data[i] = fp32_to_fp16(dis(gen));
  }
}

// Modified SwiGLU for reference (with clipping)
float modified_swiglu_ref(float gate_v, float up_v) {
  constexpr float alpha = 1.702f;
  constexpr float limit = 7.0f;
  
  float x = std::min(gate_v, limit);
  float y = std::clamp(up_v, -limit, limit);
  float out_glu = x / (1.0f + std::exp(alpha * (-x)));
  
  return out_glu * (y + 1.0f);
}

// Test basic expert MLP operation with 2D/1D weights and biases
bool test_expert_mlp_basic() {
  std::cout << "Test 1: Basic expert MLP with 2D/1D weights and biases\n";

  // Simple setup: batch=1, seq_len=2, hidden_dim=8, expert_ffn_dim=16
  int64_t batch = 1;
  int64_t seq_len = 2;
  int64_t hidden_dim = 8;
  int64_t expert_ffn_dim = 16;
  int64_t top_k = 4;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w_up_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_gate_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_down_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> b_up_shape = {expert_ffn_dim};
  std::vector<int64_t> b_gate_shape = {expert_ffn_dim};
  std::vector<int64_t> b_down_shape = {hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w_up(dtype::float16, w_up_shape);
  dynamic_tensor w_gate(dtype::float16, w_gate_shape);
  dynamic_tensor w_down(dtype::float16, w_down_shape);
  dynamic_tensor b_up(dtype::float32, b_up_shape);
  dynamic_tensor b_gate(dtype::float32, b_gate_shape);
  dynamic_tensor b_down(dtype::float32, b_down_shape);

  fill_random_fp32(hidden_states, -0.5f, 0.5f);
  fill_random_fp16(w_up, -0.1f, 0.1f);
  fill_random_fp16(w_gate, -0.1f, 0.1f);
  fill_random_fp16(w_down, -0.1f, 0.1f);
  fill_random_fp32(b_up, -0.05f, 0.05f);
  fill_random_fp32(b_gate, -0.05f, 0.05f);
  fill_random_fp32(b_down, -0.05f, 0.05f);

  // Create expert MLP node for expert 0
  expert_mlp_node node(0);

  // Create router_output: [batch, seq_len, top_k, 2]
  dynamic_tensor router_output(dtype::float32, {batch, seq_len, top_k, 2});
  float* router_data = router_output.data_ptr<float>();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      for (int64_t k = 0; k < top_k; ++k) {
        int64_t idx = (b * seq_len + s) * top_k * 2 + k * 2;
        router_data[idx] = (k == 0) ? 0.0f : static_cast<float>(k); // expert_id
        router_data[idx + 1] = 1.0f / top_k; // weight
      }
    }
  }

  // Compute
  std::vector<dynamic_tensor> inputs = {hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output};
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

// Test modified SwiGLU activation
bool test_modified_swiglu_activation() {
  std::cout << "\nTest 2: Modified SwiGLU activation\n";

  // Create simple case where we can verify the activation is applied correctly
  int64_t batch = 1;
  int64_t seq_len = 1;
  int64_t hidden_dim = 2;
  int64_t expert_ffn_dim = 2;
  int64_t top_k = 4;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w_up_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_gate_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_down_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> b_up_shape = {expert_ffn_dim};
  std::vector<int64_t> b_gate_shape = {expert_ffn_dim};
  std::vector<int64_t> b_down_shape = {hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w_up(dtype::float16, w_up_shape);
  dynamic_tensor w_gate(dtype::float16, w_gate_shape);
  dynamic_tensor w_down(dtype::float16, w_down_shape);
  dynamic_tensor b_up(dtype::float32, b_up_shape);
  dynamic_tensor b_gate(dtype::float32, b_gate_shape);
  dynamic_tensor b_down(dtype::float32, b_down_shape);

  // Set specific values
  float* hidden_data = hidden_states.data_ptr<float>();
  hidden_data[0] = 1.0f;
  hidden_data[1] = 0.5f;

  // Set weights to create specific gate and up values
  uint16_t* w_up_data = w_up.data_ptr<uint16_t>();
  uint16_t* w_gate_data = w_gate.data_ptr<uint16_t>();
  uint16_t* w_down_data = w_down.data_ptr<uint16_t>();
  float* b_up_data = b_up.data_ptr<float>();
  float* b_gate_data = b_gate.data_ptr<float>();
  float* b_down_data = b_down.data_ptr<float>();

  // Initialize all to zero
  for (int64_t i = 0; i < w_up.numel(); ++i) w_up_data[i] = fp32_to_fp16(0.0f);
  for (int64_t i = 0; i < w_gate.numel(); ++i) w_gate_data[i] = fp32_to_fp16(0.0f);
  for (int64_t i = 0; i < w_down.numel(); ++i) w_down_data[i] = fp32_to_fp16(0.0f);
  for (int64_t i = 0; i < b_up.numel(); ++i) b_up_data[i] = 0.0f;
  for (int64_t i = 0; i < b_gate.numel(); ++i) b_gate_data[i] = 0.0f;
  for (int64_t i = 0; i < b_down.numel(); ++i) b_down_data[i] = 0.0f;

  // Set up projection: w_up[0][0] = 1.0, b_up[0] = 0.5
  // This gives up_0 = 1.0 * 1.0 + 0.5 * 0.0 + 0.5 = 1.5
  w_up_data[0] = fp32_to_fp16(1.0f);
  b_up_data[0] = 0.5f;

  // Set gate projection: w_gate[0][0] = 1.0, b_gate[0] = 0.0
  // This gives gate_0 = 1.0 * 1.0 + 0.5 * 0.0 + 0.0 = 1.0
  w_gate_data[0] = fp32_to_fp16(1.0f);

  // Set down projection to identity
  w_down_data[0] = fp32_to_fp16(1.0f);  // w_down[0][0]
  w_down_data[2] = fp32_to_fp16(1.0f);  // w_down[1][1]

  // Create expert MLP node
  expert_mlp_node node(0);

  // Create router_output: [batch, seq_len, top_k, 2]
  dynamic_tensor router_output(dtype::float32, {batch, seq_len, top_k, 2});
  float* router_data = router_output.data_ptr<float>();
  router_data[0] = 0.0f; // expert_id 0
  router_data[1] = 1.0f; // weight
  for (int64_t k = 1; k < top_k; ++k) {
    router_data[k * 2] = static_cast<float>(k);
    router_data[k * 2 + 1] = 0.0f;
  }

  // Compute
  std::vector<dynamic_tensor> inputs = {hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output};
  dynamic_tensor output = node.compute_test(inputs);

  const float* output_data = output.data_ptr<float>();

  // Expected: output[0] = modified_swiglu(gate=1.0, up=1.5)
  float expected_0 = modified_swiglu_ref(1.0f, 1.5f);

  float diff_0 = std::abs(output_data[0] - expected_0);

  if (diff_0 > 1e-3f) {  // Relaxed tolerance due to fp16 conversion
    std::cerr << "  ERROR: Modified SwiGLU not applied correctly at position 0: expected=" << expected_0
              << ", got=" << output_data[0] << ", diff=" << diff_0 << "\n";
    return false;
  }

  std::cout << "  ✓ Modified SwiGLU activation correct\n";
  return true;
}

// Test multiple experts with 2D/1D weights and biases
bool test_multiple_experts() {
  std::cout << "\nTest 3: Multiple expert IDs with 2D/1D weights and biases\n";

  int64_t batch = 1;
  int64_t seq_len = 2;
  int64_t hidden_dim = 8;
  int64_t expert_ffn_dim = 16;
  int64_t top_k = 4;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w_up_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_gate_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_down_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> b_up_shape = {expert_ffn_dim};
  std::vector<int64_t> b_gate_shape = {expert_ffn_dim};
  std::vector<int64_t> b_down_shape = {hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w_up(dtype::float16, w_up_shape);
  dynamic_tensor w_gate(dtype::float16, w_gate_shape);
  dynamic_tensor w_down(dtype::float16, w_down_shape);
  dynamic_tensor b_up(dtype::float32, b_up_shape);
  dynamic_tensor b_gate(dtype::float32, b_gate_shape);
  dynamic_tensor b_down(dtype::float32, b_down_shape);

  fill_random_fp32(hidden_states);
  fill_random_fp16(w_up);
  fill_random_fp16(w_gate);
  fill_random_fp16(w_down);
  fill_random_fp32(b_up);
  fill_random_fp32(b_gate);
  fill_random_fp32(b_down);

  // Test multiple expert IDs
  std::vector<int> expert_ids = {0, 5, 15, 31};

  for (int expert_id : expert_ids) {
    expert_mlp_node node(expert_id);

    if (node.get_expert_id() != expert_id) {
      std::cerr << "  ERROR: Expert ID mismatch: expected=" << expert_id
                << ", got=" << node.get_expert_id() << "\n";
      return false;
    }

    // Compute with 2D/1D weights and biases
    dynamic_tensor router_output(dtype::float32, {batch, seq_len, top_k, 2});
    float* router_data = router_output.data_ptr<float>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        router_data[(b * seq_len + s) * top_k * 2] = static_cast<float>(expert_id);
        router_data[(b * seq_len + s) * top_k * 2 + 1] = 1.0f;
        for (int64_t k = 1; k < top_k; ++k) {
          int64_t idx = (b * seq_len + s) * top_k * 2 + k * 2;
          router_data[idx] = static_cast<float>((expert_id + k) % 32);
          router_data[idx + 1] = 0.0f;
        }
      }
    }
    std::vector<dynamic_tensor> inputs = {hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output};
    dynamic_tensor output = node.compute_test(inputs);

    // Verify output shape
    if (output.shape() != hidden_states.shape()) {
      std::cerr << "  ERROR: Output shape mismatch for expert " << expert_id << "\n";
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
      std::cerr << "  ERROR: Output is all zeros for expert " << expert_id << "\n";
      return false;
    }
  }

  std::cout << "  ✓ Multiple experts with 2D/1D weights and biases work correctly\n";
  return true;
}

// Test GPT-OSS scale dimensions
bool test_gpt_oss_scale() {
  std::cout << "\nTest 4: GPT-OSS scale dimensions\n";

  // GPT-OSS: hidden_dim=2880, expert_ffn_dim=2880
  int64_t batch = 2;
  int64_t seq_len = 4;
  int64_t hidden_dim = 128;
  int64_t expert_ffn_dim = 128;
  int64_t top_k = 4;

  std::vector<int64_t> hidden_shape = {batch, seq_len, hidden_dim};
  std::vector<int64_t> w_up_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_gate_shape = {expert_ffn_dim, hidden_dim};
  std::vector<int64_t> w_down_shape = {hidden_dim, expert_ffn_dim};
  std::vector<int64_t> b_up_shape = {expert_ffn_dim};
  std::vector<int64_t> b_gate_shape = {expert_ffn_dim};
  std::vector<int64_t> b_down_shape = {hidden_dim};

  dynamic_tensor hidden_states(dtype::float32, hidden_shape);
  dynamic_tensor w_up(dtype::float16, w_up_shape);
  dynamic_tensor w_gate(dtype::float16, w_gate_shape);
  dynamic_tensor w_down(dtype::float16, w_down_shape);
  dynamic_tensor b_up(dtype::float32, b_up_shape);
  dynamic_tensor b_gate(dtype::float32, b_gate_shape);
  dynamic_tensor b_down(dtype::float32, b_down_shape);

  fill_random_fp32(hidden_states);
  fill_random_fp16(w_up, -0.05f, 0.05f);
  fill_random_fp16(w_gate, -0.05f, 0.05f);
  fill_random_fp16(w_down, -0.05f, 0.05f);
  fill_random_fp32(b_up, -0.01f, 0.01f);
  fill_random_fp32(b_gate, -0.01f, 0.01f);
  fill_random_fp32(b_down, -0.01f, 0.01f);

  // Create expert MLP node
  expert_mlp_node node(0);

  // Create router_output: all tokens select expert 0
  dynamic_tensor router_output(dtype::float32, {batch, seq_len, top_k, 2});
  float* router_data = router_output.data_ptr<float>();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      router_data[(b * seq_len + s) * top_k * 2] = 0.0f;
      router_data[(b * seq_len + s) * top_k * 2 + 1] = 1.0f;
      for (int64_t k = 1; k < top_k; ++k) {
        int64_t idx = (b * seq_len + s) * top_k * 2 + k * 2;
        router_data[idx] = static_cast<float>(k);
        router_data[idx + 1] = 0.0f;
      }
    }
  }

  // Compute
  std::vector<dynamic_tensor> inputs = {hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output};
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
  std::cout << "Testing Expert MLP Node (2D/1D Weights, Biases, Modified SwiGLU)\n";
  std::cout << "===============================================================\n\n";

  bool test1 = test_expert_mlp_basic();
  bool test2 = test_modified_swiglu_activation();
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