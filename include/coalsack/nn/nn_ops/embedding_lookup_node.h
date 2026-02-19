#pragma once

#include "coalsack/gguf/gguf_dequant.h"
#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class embedding_lookup_node : public binary_op_node {
 public:
  embedding_lookup_node() : binary_op_node("embedding_lookup") {}

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& input_ids, const dynamic_tensor& weight) {
    return compute(input_ids, weight);
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input_ids, const dynamic_tensor& weight) override {
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

    if (input_ids.get_dtype() != dtype::int32) {
      throw std::runtime_error("embedding_lookup: input_ids must be int32");
    }

    const int32_t* ids = input_ids.data_ptr<int32_t>();
    float* out = output.data_ptr<float>();

    // Handle both fp32 and fp16 weights
    if (weight.get_dtype() == dtype::float16) {
      // fp16 weight
      const uint16_t* w_fp16 = weight.data_ptr<uint16_t>();

      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          int64_t token_id = static_cast<int64_t>(ids[b * seq_len + s]);

          if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("embedding_lookup: token_id out of range: " +
                                     std::to_string(token_id));
          }

          // Convert fp16 embedding to fp32 and copy
          const uint16_t* embedding_fp16 = w_fp16 + token_id * hidden_dim;
          float* output_pos = out + (b * seq_len + s) * hidden_dim;

          for (int64_t d = 0; d < hidden_dim; ++d) {
            output_pos[d] = fp16_to_fp32(embedding_fp16[d]);
          }
        }
      }
    } else if (weight.get_dtype() == dtype::float32) {
      // fp32 weight (existing logic)
      const float* w = weight.data_ptr<float>();

      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          int64_t token_id = static_cast<int64_t>(ids[b * seq_len + s]);

          if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("embedding_lookup: token_id out of range: " +
                                     std::to_string(token_id));
          }

          const float* embedding = w + token_id * hidden_dim;
          float* output_pos = out + (b * seq_len + s) * hidden_dim;
          std::memcpy(output_pos, embedding, hidden_dim * sizeof(float));
        }
      }
    } else {
      throw std::runtime_error("embedding_lookup: weight must be fp32 or fp16, got " +
                               dtype_name(weight.get_dtype()));
    }

    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::embedding_lookup_node, coalsack::graph_node)
