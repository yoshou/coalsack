#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class embedding_lookup_node : public binary_op_node {
 public:
  embedding_lookup_node() : binary_op_node("embedding_lookup") {}

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& input_ids, const dynamic_tensor& weight) {
    return compute(input_ids, weight);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input_ids,
                        const dynamic_tensor& weight) override {
    // input_ids: [batch, seq_len] (int32)
    // weight: [vocab_size, hidden_dim] (float32)
    // output: [batch, seq_len, hidden_dim] (float32)

    if (input_ids.ndim() != 2) {
      throw std::runtime_error("embedding_lookup: input_ids must have 2 dimensions");
    }

    if (weight.ndim() != 2) {
      throw std::runtime_error("embedding_lookup: weight must have 2 dimensions");
    }

    int64_t batch = input_ids.dim(0);
    int64_t seq_len = input_ids.dim(1);
    int64_t vocab_size = weight.dim(0);
    int64_t hidden_dim = weight.dim(1);

    // Output shape: [batch, seq_len, hidden_dim]
    std::vector<int64_t> output_shape = {batch, seq_len, hidden_dim};
    dynamic_tensor output(dtype::float32, output_shape);

    if (input_ids.get_dtype() == dtype::int32 && weight.get_dtype() == dtype::float32) {
      const int32_t* ids = input_ids.data_ptr<int32_t>();
      const float* w = weight.data_ptr<float>();
      float* out = output.data_ptr<float>();

      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          int64_t token_id = static_cast<int64_t>(ids[b * seq_len + s]);

          // Bounds check
          if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("embedding_lookup: token_id out of range: " +
                                     std::to_string(token_id));
          }

          // Copy embedding vector
          const float* embedding = w + token_id * hidden_dim;
          float* output_pos = out + (b * seq_len + s) * hidden_dim;
          std::memcpy(output_pos, embedding, hidden_dim * sizeof(float));
        }
      }
    } else {
      throw std::runtime_error(
          "embedding_lookup: only int32 input_ids and float32 weight supported");
    }

    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::embedding_lookup_node, coalsack::graph_node)
