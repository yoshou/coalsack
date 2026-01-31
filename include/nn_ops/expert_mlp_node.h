#pragma once

#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class expert_mlp_node : public variadic_op_node {
 public:
  explicit expert_mlp_node(int expert_id = 0)
      : variadic_op_node("expert_mlp", 4), expert_id_(expert_id) {}  // Support both 3 and 4 inputs

  int get_expert_id() const { return expert_id_; }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) {
    return compute(inputs);
  }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    // Support both 3-input (simple GELU) and 4-input (gated SwiGLU) architectures
    if (inputs.size() != 3 && inputs.size() != 4) {
      throw std::runtime_error("expert_mlp: expected 3 inputs (hidden_states, w1, w2) or 4 inputs (hidden_states, w_up, w_gate, w_down)");
    }

    const auto& hidden_states = inputs[0];

    if (hidden_states.ndim() != 3) {
      throw std::runtime_error("expert_mlp: hidden_states must have 3 dimensions");
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);

    // Output has same shape as input
    dynamic_tensor output(hidden_states.get_dtype(), hidden_states.shape());

    if (inputs.size() == 3) {
      // Original 2-weight architecture: gelu(w1 @ x) @ w2
      const auto& w1 = inputs[1];  // Up projection weight
      const auto& w2 = inputs[2];  // Down projection weight

      if (w1.ndim() != 2 || w2.ndim() != 2) {
        throw std::runtime_error("expert_mlp: w1 and w2 must have 2 dimensions");
      }

      int64_t expert_ffn_dim = w1.dim(1);

      if (w1.dim(0) != hidden_dim) {
        throw std::runtime_error("expert_mlp: w1 dimension mismatch");
      }

      if (w2.dim(0) != expert_ffn_dim || w2.dim(1) != hidden_dim) {
        throw std::runtime_error("expert_mlp: w2 dimension mismatch");
      }

      if (hidden_states.get_dtype() == dtype::float32) {
        compute_impl<float>(hidden_states, w1, w2, output, batch, seq_len, hidden_dim,
                           expert_ffn_dim);
      } else if (hidden_states.get_dtype() == dtype::float64) {
        compute_impl<double>(hidden_states, w1, w2, output, batch, seq_len, hidden_dim,
                            expert_ffn_dim);
      } else {
        throw std::runtime_error("expert_mlp: only float32 and float64 supported");
      }
    } else {
      // Gated architecture (4 inputs): silu(w_gate @ x) * (w_up @ x) @ w_down
      const auto& w_up = inputs[1];    // Up projection weight
      const auto& w_gate = inputs[2];  // Gate projection weight
      const auto& w_down = inputs[3];  // Down projection weight

      // Support both 2D [dim1, dim2] and 3D [num_experts, dim1, dim2] weight tensors
      bool is_3d = (w_up.ndim() == 3);

      if (!is_3d && (w_up.ndim() != 2 || w_gate.ndim() != 2 || w_down.ndim() != 2)) {
        throw std::runtime_error("expert_mlp: weights must have 2 or 3 dimensions");
      }

      if (is_3d && (w_up.ndim() != 3 || w_gate.ndim() != 3 || w_down.ndim() != 3)) {
        throw std::runtime_error("expert_mlp: all weights must have same number of dimensions");
      }

      // Extract dimensions
      int64_t expert_ffn_dim;
      if (is_3d) {
        // 3D weights: [num_experts, hidden_dim, expert_ffn_dim]
        int64_t num_experts = w_up.dim(0);
        if (expert_id_ >= num_experts) {
          throw std::runtime_error("expert_mlp: expert_id " + std::to_string(expert_id_) +
                                   " out of range (num_experts=" + std::to_string(num_experts) + ")");
        }
        expert_ffn_dim = w_up.dim(2);
      } else {
        // 2D weights: [hidden_dim, expert_ffn_dim]
        expert_ffn_dim = w_up.dim(1);
      }

      if (hidden_states.get_dtype() == dtype::float32) {
        compute_gated_impl<float>(hidden_states, w_up, w_gate, w_down, output, batch, seq_len,
                                  hidden_dim, expert_ffn_dim, is_3d);
      } else if (hidden_states.get_dtype() == dtype::float64) {
        compute_gated_impl<double>(hidden_states, w_up, w_gate, w_down, output, batch, seq_len,
                                   hidden_dim, expert_ffn_dim, is_3d);
      } else {
        throw std::runtime_error("expert_mlp: only float32 and float64 supported");
      }
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

  // Gated MLP implementation (SwiGLU): down(silu(gate(x)) * up(x))
  template <typename T>
  void compute_gated_impl(const dynamic_tensor& hidden_states, const dynamic_tensor& w_up,
                         const dynamic_tensor& w_gate, const dynamic_tensor& w_down,
                         dynamic_tensor& output, int64_t batch, int64_t seq_len,
                         int64_t hidden_dim, int64_t expert_ffn_dim, bool is_3d) {
    const T* hidden_data = hidden_states.data_ptr<T>();
    T* out_data = output.data_ptr<T>();

    // Get weight data pointers and calculate offsets for 3D tensors
    const T* w_up_data = w_up.data_ptr<T>();
    const T* w_gate_data = w_gate.data_ptr<T>();
    const T* w_down_data = w_down.data_ptr<T>();

    // If 3D, offset to the slice for this expert
    if (is_3d) {
      int64_t up_slice_size = hidden_dim * expert_ffn_dim;
      int64_t down_slice_size = expert_ffn_dim * hidden_dim;
      w_up_data += expert_id_ * up_slice_size;
      w_gate_data += expert_id_ * up_slice_size;
      w_down_data += expert_id_ * down_slice_size;
    }

    // Allocate intermediate buffers
    std::vector<T> up_proj(expert_ffn_dim);
    std::vector<T> gate_proj(expert_ffn_dim);

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const T* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        T* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        // Step 1: Up projection and gate projection
        for (int64_t i = 0; i < expert_ffn_dim; ++i) {
          T up_sum = 0;
          T gate_sum = 0;
          for (int64_t j = 0; j < hidden_dim; ++j) {
            up_sum += hidden_vec[j] * w_up_data[j * expert_ffn_dim + i];
            gate_sum += hidden_vec[j] * w_gate_data[j * expert_ffn_dim + i];
          }
          up_proj[i] = up_sum;
          gate_proj[i] = silu(gate_sum);  // Apply SiLU to gate
        }

        // Step 2: Element-wise multiply (gating)
        for (int64_t i = 0; i < expert_ffn_dim; ++i) {
          up_proj[i] *= gate_proj[i];
        }

        // Step 3: Down projection
        for (int64_t i = 0; i < hidden_dim; ++i) {
          T sum = 0;
          for (int64_t j = 0; j < expert_ffn_dim; ++j) {
            sum += up_proj[j] * w_down_data[j * hidden_dim + i];
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

  // SiLU (Swish) activation function
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  template <typename T>
  static T silu(T x) {
    return x / (static_cast<T>(1.0) + std::exp(-x));
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_mlp_node, coalsack::graph_node)
