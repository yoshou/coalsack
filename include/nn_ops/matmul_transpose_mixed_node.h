#pragma once

#include <immintrin.h>

#include "../gguf_dequant.h"
#include "../nn_op_node.h"

namespace coalsack {

// Fused MatMul with transposed weight: supports mixed fp32/fp16 inputs
// Computes: A @ B.T (PyTorch Linear layer style)
// Input A: fp32, shape [*, M, K] (activations)
// Input B: fp32 or fp16, shape [*, N, K] (weights, accessed as transposed)
// Output: fp32, shape [*, M, N]
class matmul_transpose_mixed_node : public binary_op_node {
 public:
  matmul_transpose_mixed_node() : binary_op_node("matmul_transpose_mixed") {}

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& a, const dynamic_tensor& b) {
    return compute(a, b);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() != dtype::float32) {
      throw std::runtime_error("matmul_transpose_mixed: input A must be fp32");
    }
    if (b.get_dtype() != dtype::float32 && b.get_dtype() != dtype::float16) {
      throw std::runtime_error("matmul_transpose_mixed: input B must be fp32 or fp16");
    }

    // A: [*, M, K], B: [*, N, K] -> Output: [*, M, N] = A @ B.T
    if (a.ndim() < 2 || b.ndim() < 2) {
      throw std::runtime_error("matmul_transpose_mixed: inputs must be at least 2D");
    }

    int64_t a_rows = a.dim(-2);  // M
    int64_t a_cols = a.dim(-1);  // K
    int64_t b_cols = b.dim(-2);  // N
    int64_t b_rows = b.dim(-1);  // K

    if (a_cols != b_rows) {
      throw std::runtime_error("matmul_transpose_mixed: incompatible dimensions");
    }

    std::vector<int64_t> a_batch_shape(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch_shape(b.shape().begin(), b.shape().end() - 2);

    std::vector<int64_t> batch_shape;
    if (!a_batch_shape.empty() || !b_batch_shape.empty()) {
      batch_shape = dynamic_tensor::broadcast_shape(a_batch_shape, b_batch_shape);
    }

    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(a_rows);
    output_shape.push_back(b_cols);

    dynamic_tensor output(dtype::float32, output_shape);

    int64_t batch_size = 1;
    for (auto dim : batch_shape) {
      batch_size *= dim;
    }

    const float* a_data = a.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    int64_t a_matrix_size = a_rows * a_cols;
    int64_t b_matrix_size = b_cols * b_rows;
    int64_t output_matrix_size = a_rows * b_cols;

    if (b.get_dtype() == dtype::float16) {
      // FP16 fast path: convert on-the-fly using F16C intrinsics
      const uint16_t* b_data = b.data_ptr<uint16_t>();
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        const float* a_matrix = a_data + (batch % (a.numel() / a_matrix_size)) * a_matrix_size;
        const uint16_t* b_matrix = b_data + (batch % (b.numel() / b_matrix_size)) * b_matrix_size;
        float* output_matrix = output_data + batch * output_matrix_size;
        compute_matmul_avx2_fp16b(a_matrix, b_matrix, output_matrix, a_rows, a_cols, b_cols,
                                  b_rows);
      }
    } else {
      // FP32 path
      const float* b_data = b.data_ptr<float>();
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        const float* a_matrix = a_data + (batch % (a.numel() / a_matrix_size)) * a_matrix_size;
        const float* b_matrix = b_data + (batch % (b.numel() / b_matrix_size)) * b_matrix_size;
        float* output_matrix = output_data + batch * output_matrix_size;
        compute_matmul_avx2(a_matrix, b_matrix, output_matrix, a_rows, a_cols, b_cols, b_rows);
      }
    }

    return output;
  }

 private:
  // Efficient horizontal sum using AVX2 intrinsics
  __attribute__((target("avx2"))) static inline float horizontal_sum_avx2(__m256 v) {
    // Split 256-bit into two 128-bit
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    // Add low and high
    __m128 sum128 = _mm_add_ps(low, high);
    // Horizontal add within 128-bit
    __m128 shuf = _mm_movehdup_ps(sum128);   // [1,1,3,3]
    __m128 sums = _mm_add_ps(sum128, shuf);  // [0+1, 1+1, 2+3, 3+3]
    shuf = _mm_movehl_ps(shuf, sums);        // [2+3, 3+3, ?, ?]
    sums = _mm_add_ss(sums, shuf);           // [0+1+2+3, ...]
    return _mm_cvtss_f32(sums);
  }

  // Compute A(fp32) @ B(fp32).T with AVX2 + N-tiling + K-unrolling
  // TILE_N=4: reuse a_vec across 4 B rows, 4 independent FMA chains
  // K_UNROLL=2: double accumulators to further hide FMA latency (8 chains total)
  __attribute__((target("avx2,fma"))) static void compute_matmul_avx2(
      const float* a_matrix, const float* b_matrix, float* output_matrix, int64_t a_rows,
      int64_t a_cols, int64_t b_cols, int64_t b_rows) {
    constexpr int TILE_N = 4;

    for (int64_t i = 0; i < a_rows; ++i) {
      const float* a_row = &a_matrix[i * a_cols];

      // Tiled main loop
      int64_t j = 0;
      for (; j + TILE_N <= b_cols; j += TILE_N) {
        __m256 s0a = _mm256_setzero_ps(), s0b = _mm256_setzero_ps();
        __m256 s1a = _mm256_setzero_ps(), s1b = _mm256_setzero_ps();
        __m256 s2a = _mm256_setzero_ps(), s2b = _mm256_setzero_ps();
        __m256 s3a = _mm256_setzero_ps(), s3b = _mm256_setzero_ps();

        int64_t k = 0;
        for (; k + 16 <= a_cols; k += 16) {
          __m256 a0 = _mm256_loadu_ps(&a_row[k]);
          __m256 a1 = _mm256_loadu_ps(&a_row[k + 8]);
          s0a = _mm256_fmadd_ps(a0, _mm256_loadu_ps(&b_matrix[(j + 0) * b_rows + k]), s0a);
          s0b = _mm256_fmadd_ps(a1, _mm256_loadu_ps(&b_matrix[(j + 0) * b_rows + k + 8]), s0b);
          s1a = _mm256_fmadd_ps(a0, _mm256_loadu_ps(&b_matrix[(j + 1) * b_rows + k]), s1a);
          s1b = _mm256_fmadd_ps(a1, _mm256_loadu_ps(&b_matrix[(j + 1) * b_rows + k + 8]), s1b);
          s2a = _mm256_fmadd_ps(a0, _mm256_loadu_ps(&b_matrix[(j + 2) * b_rows + k]), s2a);
          s2b = _mm256_fmadd_ps(a1, _mm256_loadu_ps(&b_matrix[(j + 2) * b_rows + k + 8]), s2b);
          s3a = _mm256_fmadd_ps(a0, _mm256_loadu_ps(&b_matrix[(j + 3) * b_rows + k]), s3a);
          s3b = _mm256_fmadd_ps(a1, _mm256_loadu_ps(&b_matrix[(j + 3) * b_rows + k + 8]), s3b);
        }

        // Merge accumulator pairs
        __m256 sum0 = _mm256_add_ps(s0a, s0b);
        __m256 sum1 = _mm256_add_ps(s1a, s1b);
        __m256 sum2 = _mm256_add_ps(s2a, s2b);
        __m256 sum3 = _mm256_add_ps(s3a, s3b);

        // Handle k remainder (8 elements at a time)
        for (; k + 8 <= a_cols; k += 8) {
          __m256 a_vec = _mm256_loadu_ps(&a_row[k]);
          sum0 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b_matrix[(j + 0) * b_rows + k]), sum0);
          sum1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b_matrix[(j + 1) * b_rows + k]), sum1);
          sum2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b_matrix[(j + 2) * b_rows + k]), sum2);
          sum3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b_matrix[(j + 3) * b_rows + k]), sum3);
        }

        float r0 = horizontal_sum_avx2(sum0);
        float r1 = horizontal_sum_avx2(sum1);
        float r2 = horizontal_sum_avx2(sum2);
        float r3 = horizontal_sum_avx2(sum3);

        // Scalar remainder
        for (; k < a_cols; ++k) {
          float a_val = a_row[k];
          r0 += a_val * b_matrix[(j + 0) * b_rows + k];
          r1 += a_val * b_matrix[(j + 1) * b_rows + k];
          r2 += a_val * b_matrix[(j + 2) * b_rows + k];
          r3 += a_val * b_matrix[(j + 3) * b_rows + k];
        }

        output_matrix[i * b_cols + j + 0] = r0;
        output_matrix[i * b_cols + j + 1] = r1;
        output_matrix[i * b_cols + j + 2] = r2;
        output_matrix[i * b_cols + j + 3] = r3;
      }

      // Remainder columns
      for (; j < b_cols; ++j) {
        __m256 sum_vec = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 8 <= a_cols; k += 8) {
          __m256 a_vec = _mm256_loadu_ps(&a_row[k]);
          sum_vec = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b_matrix[j * b_rows + k]), sum_vec);
        }
        float sum = horizontal_sum_avx2(sum_vec);
        for (; k < a_cols; ++k) {
          sum += a_row[k] * b_matrix[j * b_rows + k];
        }
        output_matrix[i * b_cols + j] = sum;
      }
    }
  }

  // Compute A(fp32) @ B(fp16).T with on-the-fly F16C conversion
  // Same tiling/unrolling strategy as fp32 path, but loads fp16 weights
  // and converts inline via _mm256_cvtph_ps, halving memory bandwidth for B.
  __attribute__((target("avx2,fma,f16c"))) static void compute_matmul_avx2_fp16b(
      const float* a_matrix, const uint16_t* b_matrix, float* output_matrix, int64_t a_rows,
      int64_t a_cols, int64_t b_cols, int64_t b_rows) {
    constexpr int TILE_N = 4;

    for (int64_t i = 0; i < a_rows; ++i) {
      const float* a_row = &a_matrix[i * a_cols];

      // Tiled main loop
      int64_t j = 0;
      for (; j + TILE_N <= b_cols; j += TILE_N) {
        __m256 s0a = _mm256_setzero_ps(), s0b = _mm256_setzero_ps();
        __m256 s1a = _mm256_setzero_ps(), s1b = _mm256_setzero_ps();
        __m256 s2a = _mm256_setzero_ps(), s2b = _mm256_setzero_ps();
        __m256 s3a = _mm256_setzero_ps(), s3b = _mm256_setzero_ps();

        int64_t k = 0;
        for (; k + 16 <= a_cols; k += 16) {
          __m256 a0 = _mm256_loadu_ps(&a_row[k]);
          __m256 a1 = _mm256_loadu_ps(&a_row[k + 8]);

          // Load fp16 and convert to fp32 on-the-fly
          s0a = _mm256_fmadd_ps(
              a0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 0) * b_rows + k])), s0a);
          s0b = _mm256_fmadd_ps(
              a1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 0) * b_rows + k + 8])),
              s0b);
          s1a = _mm256_fmadd_ps(
              a0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 1) * b_rows + k])), s1a);
          s1b = _mm256_fmadd_ps(
              a1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 1) * b_rows + k + 8])),
              s1b);
          s2a = _mm256_fmadd_ps(
              a0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 2) * b_rows + k])), s2a);
          s2b = _mm256_fmadd_ps(
              a1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 2) * b_rows + k + 8])),
              s2b);
          s3a = _mm256_fmadd_ps(
              a0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 3) * b_rows + k])), s3a);
          s3b = _mm256_fmadd_ps(
              a1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 3) * b_rows + k + 8])),
              s3b);
        }

        // Merge accumulator pairs
        __m256 sum0 = _mm256_add_ps(s0a, s0b);
        __m256 sum1 = _mm256_add_ps(s1a, s1b);
        __m256 sum2 = _mm256_add_ps(s2a, s2b);
        __m256 sum3 = _mm256_add_ps(s3a, s3b);

        // Handle k remainder (8 elements at a time)
        for (; k + 8 <= a_cols; k += 8) {
          __m256 a_vec = _mm256_loadu_ps(&a_row[k]);
          sum0 = _mm256_fmadd_ps(
              a_vec, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 0) * b_rows + k])),
              sum0);
          sum1 = _mm256_fmadd_ps(
              a_vec, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 1) * b_rows + k])),
              sum1);
          sum2 = _mm256_fmadd_ps(
              a_vec, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 2) * b_rows + k])),
              sum2);
          sum3 = _mm256_fmadd_ps(
              a_vec, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[(j + 3) * b_rows + k])),
              sum3);
        }

        float r0 = horizontal_sum_avx2(sum0);
        float r1 = horizontal_sum_avx2(sum1);
        float r2 = horizontal_sum_avx2(sum2);
        float r3 = horizontal_sum_avx2(sum3);

        // Scalar remainder
        for (; k < a_cols; ++k) {
          float a_val = a_row[k];
          float b0 = fp16_to_fp32(b_matrix[(j + 0) * b_rows + k]);
          float b1 = fp16_to_fp32(b_matrix[(j + 1) * b_rows + k]);
          float b2 = fp16_to_fp32(b_matrix[(j + 2) * b_rows + k]);
          float b3 = fp16_to_fp32(b_matrix[(j + 3) * b_rows + k]);
          r0 += a_val * b0;
          r1 += a_val * b1;
          r2 += a_val * b2;
          r3 += a_val * b3;
        }

        output_matrix[i * b_cols + j + 0] = r0;
        output_matrix[i * b_cols + j + 1] = r1;
        output_matrix[i * b_cols + j + 2] = r2;
        output_matrix[i * b_cols + j + 3] = r3;
      }

      // Remainder columns
      for (; j < b_cols; ++j) {
        __m256 sum_vec = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 8 <= a_cols; k += 8) {
          __m256 a_vec = _mm256_loadu_ps(&a_row[k]);
          sum_vec = _mm256_fmadd_ps(
              a_vec, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)&b_matrix[j * b_rows + k])),
              sum_vec);
        }
        float sum = horizontal_sum_avx2(sum_vec);
        for (; k < a_cols; ++k) {
          sum += a_row[k] * fp16_to_fp32(b_matrix[j * b_rows + k]);
        }
        output_matrix[i * b_cols + j] = sum;
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::matmul_transpose_mixed_node, coalsack::graph_node)
