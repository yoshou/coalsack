#pragma once

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
    // Convert b to fp32 if needed
    dynamic_tensor b_fp32 = to_fp32(b);

    // Now both inputs are fp32
    if (a.get_dtype() != dtype::float32 || b_fp32.get_dtype() != dtype::float32) {
      throw std::runtime_error("matmul_transpose_mixed: inputs must be fp32 or fp16");
    }

    // Matmul with B accessed as transposed (no actual transpose operation)
    // A: [*, M, K], B: [*, N, K]
    // Output: [*, M, N] = A @ B.T
    if (a.ndim() < 2 || b_fp32.ndim() < 2) {
      throw std::runtime_error("matmul_transpose_mixed: inputs must be at least 2D");
    }

    int64_t a_rows = a.dim(-2);      // M (output rows)
    int64_t a_cols = a.dim(-1);      // K (inner dimension)
    int64_t b_cols = b_fp32.dim(-2); // N (output columns)
    int64_t b_rows = b_fp32.dim(-1); // K (inner dimension, must equal a_cols)

    if (a_cols != b_rows) {
      throw std::runtime_error("matmul_transpose_mixed: incompatible dimensions");
    }

    std::vector<int64_t> a_batch_shape(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch_shape(b_fp32.shape().begin(), b_fp32.shape().end() - 2);

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
    const float* b_data = b_fp32.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    int64_t a_matrix_size = a_rows * a_cols;
    int64_t b_matrix_size = b_cols * b_rows; // b_cols * b_rows = N * K
    int64_t output_matrix_size = a_rows * b_cols;

    for (int64_t batch = 0; batch < batch_size; ++batch) {
      int64_t a_batch_idx = batch;
      int64_t b_batch_idx = batch;

      const float* a_matrix = a_data + (a_batch_idx % (a.numel() / a_matrix_size)) * a_matrix_size;
      const float* b_matrix =
          b_data + (b_batch_idx % (b_fp32.numel() / b_matrix_size)) * b_matrix_size;
      float* output_matrix = output_data + batch * output_matrix_size;

      // Compute A @ B.T
      for (int64_t i = 0; i < a_rows; ++i) {
        for (int64_t j = 0; j < b_cols; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < a_cols; ++k) {
            // A[i, k] * B.T[k, j] = A[i, k] * B[j, k]
            sum += a_matrix[i * a_cols + k] * b_matrix[j * b_rows + k];
          }
          output_matrix[i * b_cols + j] = sum;
        }
      }
    }

    return output;
  }

 private:
  // Convert fp16 → fp32 (or pass through fp32)
  static dynamic_tensor to_fp32(const dynamic_tensor& input) {
    if (input.get_dtype() == dtype::float32) {
      return input;  // Already fp32
    }

    if (input.get_dtype() != dtype::float16) {
      throw std::runtime_error("matmul_transpose_mixed: only fp32 and fp16 supported, got " +
                               dtype_name(input.get_dtype()));
    }

    // Convert fp16 → fp32
    dynamic_tensor output(dtype::float32, input.shape());

    const uint16_t* src = input.data_ptr<uint16_t>();
    float* dst = output.data_ptr<float>();

    for (int64_t i = 0; i < input.numel(); ++i) {
      dst[i] = fp16_to_fp32(src[i]);
    }

    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::matmul_transpose_mixed_node, coalsack::graph_node)
