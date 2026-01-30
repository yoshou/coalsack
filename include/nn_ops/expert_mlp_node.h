#pragma once

#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class expert_mlp_node : public variadic_op_node {
 public:
  explicit expert_mlp_node(int expert_id = 0)
      : variadic_op_node("expert_mlp", 3), expert_id_(expert_id) {}

  int get_expert_id() const { return expert_id_; }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) {
    return compute(inputs);
  }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    if (inputs.size() != 3) {
      throw std::runtime_error("expert_mlp: expected 3 inputs (hidden_states, w1, w2)");
    }

    const auto& hidden_states = inputs[0];
    const auto& w1 = inputs[1];  // Up projection weight
    const auto& w2 = inputs[2];  // Down projection weight

    // hidden_states: [batch, seq_len, hidden_dim]
    // w1: [hidden_dim, expert_ffn_dim]
    // w2: [expert_ffn_dim, hidden_dim]

    if (hidden_states.ndim() != 3) {
      throw std::runtime_error("expert_mlp: hidden_states must have 3 dimensions");
    }

    if (w1.ndim() != 2 || w2.ndim() != 2) {
      throw std::runtime_error("expert_mlp: w1 and w2 must have 2 dimensions");
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);
    int64_t expert_ffn_dim = w1.dim(1);

    // Verify dimensions
    if (w1.dim(0) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w1 dimension mismatch");
    }

    if (w2.dim(0) != expert_ffn_dim || w2.dim(1) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w2 dimension mismatch");
    }

    // Output has same shape as input
    dynamic_tensor output(hidden_states.get_dtype(), hidden_states.shape());

    if (hidden_states.get_dtype() == dtype::float32) {
      compute_impl<float>(hidden_states, w1, w2, output, batch, seq_len, hidden_dim,
                         expert_ffn_dim);
    } else if (hidden_states.get_dtype() == dtype::float64) {
      compute_impl<double>(hidden_states, w1, w2, output, batch, seq_len, hidden_dim,
                          expert_ffn_dim);
    } else {
      throw std::runtime_error("expert_mlp: only float32 and float64 supported");
    }

    return output;
  }

 private:
  int expert_id_;

  template <typename T>
  void compute_impl(const dynamic_tensor& hidden_states, const dynamic_tensor& w1,
                    const dynamic_tensor& w2, dynamic_tensor& output, int64_t batch,
                    int64_t seq_len, int64_t hidden_dim, int64_t expert_ffn_dim) {
    const T* hidden_data = hidden_states.data_ptr<T>();
    const T* w1_data = w1.data_ptr<T>();
    const T* w2_data = w2.data_ptr<T>();
    T* out_data = output.data_ptr<T>();

    // Allocate intermediate buffer for up projection
    std::vector<T> intermediate(expert_ffn_dim);

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const T* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        T* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        // Step 1: Up projection with GELU activation
        // intermediate = gelu(hidden_states @ w1)
        for (int64_t i = 0; i < expert_ffn_dim; ++i) {
          T sum = 0;
          for (int64_t j = 0; j < hidden_dim; ++j) {
            sum += hidden_vec[j] * w1_data[j * expert_ffn_dim + i];
          }
          // Apply GELU activation
          intermediate[i] = gelu(sum);
        }

        // Step 2: Down projection
        // output = intermediate @ w2
        for (int64_t i = 0; i < hidden_dim; ++i) {
          T sum = 0;
          for (int64_t j = 0; j < expert_ffn_dim; ++j) {
            sum += intermediate[j] * w2_data[j * hidden_dim + i];
          }
          output_vec[i] = sum;
        }
      }
    }
  }

  // GELU activation function
  // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  template <typename T>
  static T gelu(T x) {
    constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);  // sqrt(2/pi)
    constexpr T coeff = static_cast<T>(0.044715);

    T x3 = x * x * x;
    T inner = sqrt_2_over_pi * (x + coeff * x3);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + std::tanh(inner));
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_mlp_node, coalsack::graph_node)
