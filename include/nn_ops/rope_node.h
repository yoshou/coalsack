#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <iostream>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

// Rotary Position Embedding (RoPE) node for transformer models
// Applies rotary positional embeddings to query/key tensors in attention layers
// 
// Architecture:
// - Input format: [batch, num_heads, seq_len, head_dim] in FLOAT32/FLOAT64
// - Optional position_ids: [batch, seq_len] or [seq_len] in INT32/INT64
// - Output format: Same shape as input with RoPE applied
// - Rotation method: NeoX-style (rotate first half with second half)
// - Scaling support: None, Linear, YaRN (Yet another RoPE extensioN)
// 
// RoPE Scaling Strategies:
// - "none": Standard RoPE without scaling
// - "linear": Simple frequency scaling by 1/scaling_factor
// - "yarn": Advanced scaling with interpolation and extrapolation mixing
//           Supports corr-dims for selective dimension scaling
class rope_node : public variadic_op_node {
 public:
  rope_node()
      : variadic_op_node("rope", 1),  // Default: 1 required input (can accept 2)
        head_dim_(0),
        base_(10000.0f),
        scaling_factor_(1.0f),
        scaling_type_("none"),
        n_ctx_orig_(0),
        yarn_ext_factor_(-1.0f),
        yarn_attn_factor_(1.0f),
        yarn_beta_fast_(32.0f),
        yarn_beta_slow_(1.0f) {}

  // Simplified configuration (backward compatible)
  // If scaling_factor is 1.0, uses "none" (no scaling), otherwise "linear"
  void set_config(int64_t head_dim, int64_t max_seq_len, float base = 10000.0f,
                  float scaling_factor = 1.0f) {
    std::string scaling_type = (std::abs(scaling_factor - 1.0f) < 1e-6f) ? "none" : "linear";
    set_config(head_dim, max_seq_len, base, scaling_factor, scaling_type, /*n_ctx_orig=*/0);
  }

  // Advanced configuration for RoPE with scaling support
  // 
  // Parameters:
  // - head_dim: Dimension per attention head
  // - max_seq_len: Maximum sequence length limit
  // - base: Base frequency for rotary embeddings (default: 10000.0)
  // - scaling_factor: Scaling factor for frequency adjustment
  // - scaling_type: "none" | "linear" | "yarn"
  // - n_ctx_orig: Original context length for YaRN corr-dims (0 disables)
  // - yarn_ext_factor: YaRN extrapolation factor (-1 = auto)
  // - yarn_attn_factor: YaRN attention scaling factor
  // - yarn_beta_fast/slow: YaRN frequency bounds for corr-dims
  void set_config(int64_t head_dim, int64_t max_seq_len, float base, float scaling_factor,
                  const std::string& scaling_type, int64_t n_ctx_orig,
                  float yarn_ext_factor = -1.0f, float yarn_attn_factor = 1.0f,
                  float yarn_beta_fast = 32.0f, float yarn_beta_slow = 1.0f) {
    head_dim_ = head_dim;
    base_ = base;
    scaling_factor_ = scaling_factor;
    scaling_type_ = scaling_type;
    max_seq_len_limit_ = max_seq_len;
    n_ctx_orig_ = n_ctx_orig;
    yarn_ext_factor_ = yarn_ext_factor;
    yarn_attn_factor_ = yarn_attn_factor;
    yarn_beta_fast_ = yarn_beta_fast;
    yarn_beta_slow_ = yarn_beta_slow;
  }

  // Public wrapper for testing
  dynamic_tensor compute_test(const dynamic_tensor& input) { 
    return compute({input}); 
  }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    if (inputs.empty() || inputs.size() > 2) {
      throw std::runtime_error("rope: expected 1 or 2 inputs (tensor, optional position_ids)");
    }
    
    const auto& input = inputs[0];
    const auto& shape = input.shape();
    
    // Validate input shape: must be 4D [batch, num_heads, seq_len, head_dim]
    if (shape.size() != 4) {
      throw std::runtime_error("rope: input must be 4D [batch, num_heads, seq_len, head_dim], got " + 
                               std::to_string(shape.size()) + "D");
    }

    int64_t batch = shape[0];
    int64_t num_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t head_dim = shape[3];


    // Parse optional position_ids
    std::vector<int64_t> position_ids;
    if (inputs.size() == 2) {
      const auto& pos_tensor = inputs[1];
      // Debug prints
      // std::cout << "RoPE[" << name() << "] Input2 Dims: " << pos_tensor.ndim() << "\n";
      
      if (pos_tensor.ndim() != 1 && pos_tensor.ndim() != 2) {
        throw std::runtime_error("rope: position_ids must be 1D [seq_len] or 2D [batch, seq_len]");
      }
      
      int64_t pos_len = pos_tensor.ndim() == 1 ? pos_tensor.dim(0) : pos_tensor.dim(1);
      if (pos_len != seq_len) {
         // Debug info before throwing
         std::cout << "RoPE Error: pos_len=" << pos_len << ", seq_len=" << seq_len << "\n";
        throw std::runtime_error("rope: position_ids length mismatch");
      }
      
      position_ids.resize(seq_len);
      if (pos_tensor.get_dtype() == dtype::int32) {
        const int32_t* pos_data = pos_tensor.data_ptr<int32_t>();
        for (int64_t i = 0; i < seq_len; ++i) {
          position_ids[i] = static_cast<int64_t>(pos_data[i]);
        }
      } else if (pos_tensor.get_dtype() == dtype::int64) {
        const int64_t* pos_data = pos_tensor.data_ptr<int64_t>();
        for (int64_t i = 0; i < seq_len; ++i) {
          position_ids[i] = pos_data[i];
        }
      } else {
        throw std::runtime_error("rope: position_ids must be int32 or int64");
      }
      
      // Debug print first few positions
      if (seq_len > 0) {
          // std::cout << "RoPE[" << name() << "] Positions: " << position_ids[0] << (seq_len > 1 ? ", " + std::to_string(position_ids[1]) : "") << "...\n";
      }

    } else {
      // Default: sequential positions 0, 1, 2, ...
      position_ids.resize(seq_len);
      for (int64_t i = 0; i < seq_len; ++i) {
        position_ids[i] = i;
      }
    }
    // Validate head_dim matches configuration
    if (head_dim != head_dim_) {
      throw std::runtime_error("rope: head_dim mismatch, expected " + std::to_string(head_dim_) +
                               ", got " + std::to_string(head_dim) +
                               " (input shape: [" + std::to_string(batch) + ", " + std::to_string(num_heads) +
                               ", " + std::to_string(seq_len) + ", " + std::to_string(head_dim) + "])");
    }

    // Validate sequence length does not exceed limit
    if (max_seq_len_limit_ > 0 && seq_len > max_seq_len_limit_) {
      throw std::runtime_error("rope: seq_len " + std::to_string(seq_len) + 
                               " exceeds max_seq_len_limit " + std::to_string(max_seq_len_limit_));
    }

    // Allocate output tensor with same shape and dtype as input
    dynamic_tensor output(input.get_dtype(), shape);

    // Dispatch to type-specific implementation
    if (input.get_dtype() == dtype::float32) {
      compute_impl<float>(input, output, batch, num_heads, seq_len, head_dim, position_ids);
    } else if (input.get_dtype() == dtype::float64) {
      compute_impl<double>(input, output, batch, num_heads, seq_len, head_dim, position_ids);
    } else {
      throw std::runtime_error("rope: only FLOAT32 and FLOAT64 supported");
    }

    return output;
  }

 private:
  // Configuration parameters
  int64_t head_dim_;
  float base_;
  float scaling_factor_;
  std::string scaling_type_;
  int64_t max_seq_len_limit_ = 0;
  int64_t n_ctx_orig_;
  
  // YaRN-specific parameters
  float yarn_ext_factor_;
  float yarn_attn_factor_;
  float yarn_beta_fast_;
  float yarn_beta_slow_;

  // YaRN helper: Compute ramp mixing factor for corr-dims interpolation
  // Returns smooth transition weight in [0, 1] based on dimension index
  static float yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
  }

  // YaRN helper: Compute critical dimension for frequency adjustment
  // Determines which dimensions need interpolation vs extrapolation
  static float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    constexpr float kPi = 3.14159265358979323846f;
    return n_dims * std::log(float(n_ctx_orig) / (n_rot * 2.0f * kPi)) / (2.0f * std::log(base));
  }

  // YaRN helper: Compute dimension range [low, high] for corr-dims scaling
  // Based on beta_fast (high freq) and beta_slow (low freq) boundaries
  static void yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
                             float dims_out[2]) {
    const float start = std::floor(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = std::ceil(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims_out[0] = std::max(0.0f, start);
    dims_out[1] = std::min(float(n_dims - 1), end);
  }

  // Compute theta and magnitude scale for given position and dimension
  // Supports multiple scaling strategies: none, linear, YaRN
  void compute_theta_mscale(int64_t pos, int64_t dim_idx, float& theta_out, float& mscale_out) const {
    // Determine scaling strategy and frequency scale factor
    const bool use_yarn = (scaling_type_ == "yarn");
    const bool use_linear = (scaling_type_ == "linear");
    const float freq_scale = (scaling_type_ == "none" || scaling_factor_ == 0.0f) ? 1.0f : (1.0f / scaling_factor_);

    // Configure YaRN extrapolation factor (auto if negative)
    float ext_factor = yarn_ext_factor_;
    if (ext_factor < 0.0f) {
      ext_factor = use_yarn ? 1.0f : 0.0f;
    }

    // Configure YaRN attention scaling factor with magnitude correction
    float attn_factor = yarn_attn_factor_;
    if (use_yarn && ext_factor != 0.0f) {
      const float factor = 1.0f / freq_scale;
      const float log_term = (factor <= 1.0f) ? 0.0f : std::log(factor);
      const float mscale = (factor <= 1.0f) ? 1.0f : (0.1f * 1.0f * log_term + 1.0f);
      attn_factor = mscale;
      attn_factor *= 1.0f / (1.0f + 0.1f * log_term);
    }

    // Compute YaRN corr-dims range if enabled
    float corr_dims[2] = {0.0f, 0.0f};
    const bool has_corr_dims = use_yarn && n_ctx_orig_ > 0;
    if (has_corr_dims) {
      yarn_corr_dims(int(head_dim_), int(n_ctx_orig_), base_, yarn_beta_fast_, yarn_beta_slow_, corr_dims);
    }

    // Base theta for extrapolation: pos / (base^(2*dim_idx/head_dim))
    const float theta_extrap = float(pos) * std::pow(base_, -2.0f * float(dim_idx) / float(head_dim_));

    float theta = theta_extrap;
    float mscale = 1.0f;

    // Apply scaling strategy
    if (use_yarn) {
      const float theta_interp = freq_scale * theta_extrap;
      theta = theta_interp;

      if (ext_factor != 0.0f && has_corr_dims) {
        // Selective scaling with smooth transition
        const int i0 = int(2 * dim_idx);
        const float ramp_mix = yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale = attn_factor * (1.0f + 0.1f * std::log(1.0f / freq_scale));
      } else if (ext_factor != 0.0f) {
        mscale = attn_factor * (1.0f + 0.1f * std::log(1.0f / freq_scale));
      }
    } else if (use_linear) {
      theta = freq_scale * theta_extrap;
    }

    theta_out = theta;
    mscale_out = mscale;
  }

  // Apply RoPE rotation to input tensor
  // NeoX-style: rotates pairs (x[i], x[i+half_dim]) for i in [0, half_dim)
  // Computes cos/sin on-the-fly for simplicity
  template <typename T>
  void compute_impl(const dynamic_tensor& input, dynamic_tensor& output, int64_t batch,
                    int64_t num_heads, int64_t seq_len, int64_t head_dim,
                    const std::vector<int64_t>& position_ids) {
    const T* x = input.data_ptr<T>();
    T* out = output.data_ptr<T>();

    int64_t half_dim = head_dim / 2;

    // Apply rotation for each batch, head, and position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t pos = 0; pos < seq_len; ++pos) {
          // Get absolute position from position_ids
          int64_t absolute_pos = position_ids[pos];
          
          // Rotate each dimension pair with on-the-fly cos/sin computation
          for (int64_t d = 0; d < half_dim; ++d) {
            // Calculate index for input
            int64_t base_idx = ((b * num_heads + h) * seq_len + pos) * head_dim;

            // Get dimension pair to rotate: (x[d], x[d+half_dim])
            // NeoX-style: split first half and second half
            T x0 = x[base_idx + d];
            T x1 = x[base_idx + d + half_dim];

            // Compute theta and magnitude scale for this position and dimension
            float theta, mscale;
            compute_theta_mscale(absolute_pos, d, theta, mscale);

            // Compute cos/sin values
            T cos_val = static_cast<T>(std::cos(theta) * mscale);
            T sin_val = static_cast<T>(std::sin(theta) * mscale);

            // Apply 2D rotation matrix: [cos -sin; sin cos] @ [x0; x1]
            out[base_idx + d] = x0 * cos_val - x1 * sin_val;
            out[base_idx + d + half_dim] = x0 * sin_val + x1 * cos_val;
          }

          // Handle odd head_dim: copy unpaired last element
          if (head_dim % 2 != 0) {
            int64_t base_idx = ((b * num_heads + h) * seq_len + pos) * head_dim;
            out[base_idx + head_dim - 1] = x[base_idx + head_dim - 1];
          }
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::rope_node, coalsack::graph_node)
