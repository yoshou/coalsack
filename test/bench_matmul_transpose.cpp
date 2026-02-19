#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "coalsack/gguf/gguf_dequant.h"
#include "coalsack/nn/nn_ops/matmul_transpose_mixed_node.h"

using namespace coalsack;

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
  int64_t M;
  int64_t N;
  int64_t K;
  bool fp16_b;
  int warmup_iters;
  int bench_iters;
};

// =============================================================================
// Benchmark runner
// =============================================================================

static void run_benchmark(const BenchConfig& config) {
  std::cout << "\n=== " << config.name << " (" << config.M << "x" << config.N << "x" << config.K
            << ", B=" << (config.fp16_b ? "fp16" : "fp32") << ") ===\n";

  // Prepare input tensors
  dynamic_tensor a(dtype::float32, {config.M, config.K});
  fill_random_fp32(a.data_ptr<float>(), a.numel(), -1.0f, 1.0f, 42);

  dynamic_tensor b_input(dtype::float32, {config.N, config.K});
  fill_random_fp32(b_input.data_ptr<float>(), b_input.numel(), -1.0f, 1.0f, 123);

  // Convert B to fp16 if requested
  if (config.fp16_b) {
    dynamic_tensor b_fp16(dtype::float16, {config.N, config.K});
    const float* src = b_input.data_ptr<float>();
    uint16_t* dst = b_fp16.data_ptr<uint16_t>();
    for (int64_t i = 0; i < b_input.numel(); ++i) {
      dst[i] = fp32_to_fp16(src[i]);
    }
    b_input = b_fp16;
  }

  auto node = std::make_shared<matmul_transpose_mixed_node>();

  // Sanity check
  auto sanity = node->compute_test(a, b_input);
  bool has_nonzero = false;
  const float* s = sanity.data_ptr<float>();
  for (int64_t i = 0; i < sanity.numel(); ++i) {
    if (std::abs(s[i]) > 1e-8f) {
      has_nonzero = true;
      break;
    }
  }
  std::cout << "  Sanity check: " << (has_nonzero ? "PASS" : "FAIL") << "\n";

  // Warmup
  for (int i = 0; i < config.warmup_iters; ++i) {
    auto result = node->compute_test(a, b_input);
  }

  // Measure
  std::vector<double> times(config.bench_iters);
  for (int i = 0; i < config.bench_iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = node->compute_test(a, b_input);
    auto end = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  }

  double mean = std::accumulate(times.begin(), times.end(), 0.0) / config.bench_iters;
  double sq_sum = 0.0;
  for (auto t : times) sq_sum += (t - mean) * (t - mean);
  double stddev = std::sqrt(sq_sum / config.bench_iters);

  // FLOP count: 2 * M * N * K
  double gflops = (2.0 * config.M * config.N * config.K) / (mean * 1e6);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Mean:   " << mean << " ms\n";
  std::cout << "  Stddev: " << stddev << " ms\n";
  std::cout << "  GFLOPS: " << std::setprecision(4) << gflops << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
  std::cout << "=== matmul_transpose_mixed Benchmark ===\n";

  std::vector<BenchConfig> configs = {
      // Small (debug)
      {"Small", 32, 32, 128, false, 10, 100},

      // GPT-2 scale (768)
      {"GPT-2 fp32", 1, 768, 768, false, 5, 50},
      {"GPT-2 fp16", 1, 768, 768, true, 5, 50},
      {"GPT-2 batch fp32", 8, 768, 768, false, 5, 50},
      {"GPT-2 batch fp16", 8, 768, 768, true, 5, 50},

      // Model scale (2880)
      {"Model fp32", 1, 2880, 2880, false, 3, 20},
      {"Model fp16", 1, 2880, 2880, true, 3, 20},
      {"Model batch fp32", 4, 2880, 2880, false, 3, 20},
      {"Model batch fp16", 4, 2880, 2880, true, 3, 20},

      // GPT-3 scale (4096)
      {"GPT-3 fp32", 1, 4096, 4096, false, 3, 20},
      {"GPT-3 fp16", 1, 4096, 4096, true, 3, 20},
      {"GPT-3 batch fp32", 4, 4096, 4096, false, 3, 20},
      {"GPT-3 batch fp16", 4, 4096, 4096, true, 3, 20},

      // XLarge (8192)
      {"XLarge fp32", 1, 8192, 8192, false, 2, 10},
      {"XLarge fp16", 1, 8192, 8192, true, 2, 10},
  };

  for (const auto& config : configs) {
    run_benchmark(config);
  }

  std::cout << "\nDone.\n";
  return 0;
}
