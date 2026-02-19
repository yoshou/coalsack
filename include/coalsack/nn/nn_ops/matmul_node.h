#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class matmul_node : public binary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() != dtype::float32 || b.get_dtype() != dtype::float32) {
      throw std::runtime_error("matmul: only float32 supported");
    }

    if (a.ndim() < 2 || b.ndim() < 2) {
      throw std::runtime_error("matmul: inputs must be at least 2D");
    }

    int64_t a_rows = a.dim(-2);
    int64_t a_cols = a.dim(-1);
    int64_t b_rows = b.dim(-2);
    int64_t b_cols = b.dim(-1);

    if (a_cols != b_rows) {
      throw std::runtime_error("matmul: incompatible dimensions");
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
    const float* b_data = b.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    int64_t a_matrix_size = a_rows * a_cols;
    int64_t b_matrix_size = b_rows * b_cols;
    int64_t output_matrix_size = a_rows * b_cols;

    for (int64_t batch = 0; batch < batch_size; ++batch) {
      int64_t a_batch_idx = batch;
      int64_t b_batch_idx = batch;

      const float* a_matrix = a_data + (a_batch_idx % (a.numel() / a_matrix_size)) * a_matrix_size;
      const float* b_matrix = b_data + (b_batch_idx % (b.numel() / b_matrix_size)) * b_matrix_size;
      float* output_matrix = output_data + batch * output_matrix_size;

      for (int64_t i = 0; i < a_rows; ++i) {
        for (int64_t j = 0; j < b_cols; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < a_cols; ++k) {
            sum += a_matrix[i * a_cols + k] * b_matrix[k * b_cols + j];
          }
          output_matrix[i * b_cols + j] = sum;
        }
      }
    }

    return output;
  }

 public:
  matmul_node() : binary_op_node("matmul") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::matmul_node, coalsack::graph_node)
