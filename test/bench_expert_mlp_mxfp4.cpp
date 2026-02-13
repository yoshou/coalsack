#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "gguf_dequant.h"
#include "nn_nodes.h"

using namespace coalsack;

// =============================================================================
// FP32 -> MXFP4 quantization helper (for test data generation)
// =============================================================================

static uint8_t find_e8m0_exponent(const float* values, int count) {
  float max_abs = 0.0f;
  for (int i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::abs(values[i]));
  }
  if (max_abs == 0.0f) {
    return 127;
  }
  float log2_val = std::log2(max_abs / 6.0f);
  int e = 128 + static_cast<int>(std::ceil(log2_val));
  e = std::clamp(e, 2, 254);
  return static_cast<uint8_t>(e);
}

static uint8_t find_closest_mxfp4(float value, float scale) {
  if (scale == 0.0f) return 0;
  float scaled = value / scale;
  static constexpr float abs_vals[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f};
  float abs_scaled = std::abs(scaled);
  int best_idx = 0;
  float best_dist = abs_scaled;
  for (int i = 1; i < 8; ++i) {
    float dist = std::abs(abs_scaled - abs_vals[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best_idx = i;
    }
  }
  if (scaled < 0.0f && best_idx > 0) {
    best_idx += 8;
  }
  return static_cast<uint8_t>(best_idx);
}

static void quantize_fp32_to_mxfp4(const float* src, uint8_t* dst, int64_t k) {
  constexpr int QK = 32;
  constexpr int BLOCK_BYTES = 17;
  int64_t num_blocks = k / QK;
  for (int64_t b = 0; b < num_blocks; ++b) {
    const float* block_src = src + b * QK;
    uint8_t* block_dst = dst + b * BLOCK_BYTES;
    uint8_t e = find_e8m0_exponent(block_src, QK);
    block_dst[0] = e;
    float scale = e8m0_to_fp32_half(e);
    uint8_t indices[QK];
    for (int i = 0; i < QK; ++i) {
      indices[i] = find_closest_mxfp4(block_src[i], scale);
    }
    for (int j = 0; j < 16; ++j) {
      block_dst[1 + j] = (indices[j] & 0x0F) | ((indices[j + 16] & 0x0F) << 4);
    }
  }
}

// =============================================================================
// Helper functions
// =============================================================================

static void fill_random_fp32(float* data, int64_t count, float min_val, float max_val,
                              uint32_t seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(min_val, max_val);
  for (int64_t i = 0; i < count; ++i) {
    data[i] = dis(gen);
  }
}

struct BenchConfig {
  std::string name;
  int64_t hidden_dim;
  int64_t expert_ffn_dim;
  int warmup_iters;
  int bench_iters;
};

// =============================================================================
// Benchmark runner
// =============================================================================

static void run_benchmark(const BenchConfig& config) {
  std::cout << "\n=== " << config.name << " (" << config.hidden_dim << "x" << config.expert_ffn_dim
            << ") ===\n";

  const int64_t hidden_dim = config.hidden_dim;
  const int64_t expert_ffn_dim = config.expert_ffn_dim;

  // Allocate FP32 hidden states and biases
  std::vector<float> hidden_vec(hidden_dim);
  std::vector<float> b_up(expert_ffn_dim);
  std::vector<float> b_gate(expert_ffn_dim);
  std::vector<float> b_down(hidden_dim);
  std::vector<float> output(hidden_dim);

  fill_random_fp32(hidden_vec.data(), hidden_dim, -0.5f, 0.5f, 42);
  fill_random_fp32(b_up.data(), expert_ffn_dim, -0.05f, 0.05f, 43);
  fill_random_fp32(b_gate.data(), expert_ffn_dim, -0.05f, 0.05f, 44);
  fill_random_fp32(b_down.data(), hidden_dim, -0.05f, 0.05f, 45);

  // Create FP32 weight data, then quantize to MXFP4
  int64_t up_elements = expert_ffn_dim * hidden_dim;
  int64_t down_elements = hidden_dim * expert_ffn_dim;

  std::vector<float> w_up_fp32(up_elements);
  std::vector<float> w_gate_fp32(up_elements);
  std::vector<float> w_down_fp32(down_elements);

  fill_random_fp32(w_up_fp32.data(), up_elements, -0.1f, 0.1f, 100);
  fill_random_fp32(w_gate_fp32.data(), up_elements, -0.1f, 0.1f, 200);
  fill_random_fp32(w_down_fp32.data(), down_elements, -0.1f, 0.1f, 300);

  // Quantize to MXFP4
  constexpr int BLOCK_BYTES = 17;
  constexpr int QK = 32;
  int64_t up_blocks = up_elements / QK;
  int64_t down_blocks = down_elements / QK;

  std::vector<uint8_t> w_up_mxfp4(up_blocks * BLOCK_BYTES);
  std::vector<uint8_t> w_gate_mxfp4(up_blocks * BLOCK_BYTES);
  std::vector<uint8_t> w_down_mxfp4(down_blocks * BLOCK_BYTES);

  quantize_fp32_to_mxfp4(w_up_fp32.data(), w_up_mxfp4.data(), up_elements);
  quantize_fp32_to_mxfp4(w_gate_fp32.data(), w_gate_mxfp4.data(), up_elements);
  quantize_fp32_to_mxfp4(w_down_fp32.data(), w_down_mxfp4.data(), down_elements);

  // Create expert_mlp_node instance
  expert_mlp_node node(0);

  // FLOP count: 6 * hidden_dim * expert_ffn_dim (up + gate + down multiply-accumulate)
  double flops = 6.0 * hidden_dim * expert_ffn_dim;

  // Sanity check: verify output is not all zeros
  node.compute_test_mxfp4(hidden_vec.data(), w_up_mxfp4.data(), w_gate_mxfp4.data(),
                           w_down_mxfp4.data(), b_up.data(), b_gate.data(), b_down.data(),
                           output.data(), hidden_dim, expert_ffn_dim);
  bool has_nonzero = false;
  for (int64_t i = 0; i < hidden_dim; ++i) {
    if (std::abs(output[i]) > 1e-8f) {
      has_nonzero = true;
      break;
    }
  }
  std::cout << "  Sanity check: " << (has_nonzero ? "PASS" : "FAIL") << "\n";

  // Warmup
  for (int i = 0; i < config.warmup_iters; ++i) {
    node.compute_test_mxfp4(hidden_vec.data(), w_up_mxfp4.data(), w_gate_mxfp4.data(),
                             w_down_mxfp4.data(), b_up.data(), b_gate.data(), b_down.data(),
                             output.data(), hidden_dim, expert_ffn_dim);
  }

  // Measure
  std::vector<double> times(config.bench_iters);
  for (int i = 0; i < config.bench_iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    node.compute_test_mxfp4(hidden_vec.data(), w_up_mxfp4.data(), w_gate_mxfp4.data(),
                             w_down_mxfp4.data(), b_up.data(), b_gate.data(), b_down.data(),
                             output.data(), hidden_dim, expert_ffn_dim);
    auto end = std::chrono::high_resolution_clock::now();
    times[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  }

  double mean = std::accumulate(times.begin(), times.end(), 0.0) / config.bench_iters;
  double sq_sum = 0.0;
  for (auto t : times) sq_sum += (t - mean) * (t - mean);
  double stddev = std::sqrt(sq_sum / config.bench_iters);
  double gflops = flops / (mean * 1e6);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Mean:   " << mean << " ms\n";
  std::cout << "  Stddev: " << stddev << " ms\n";
  std::cout << "  GFLOPS: " << std::setprecision(4) << gflops << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
  std::cout << "=== Expert MLP MXFP4 Benchmark ===\n";

  std::vector<BenchConfig> configs = {
      {"Small", 256, 256, 10, 100},
      {"Medium", 1024, 1024, 10, 50},
      {"Full (GPT-OSS)", 2880, 2880, 5, 20},
  };

  for (const auto& config : configs) {
    run_benchmark(config);
  }

  std::cout << "\nDone.\n";
  return 0;
}
