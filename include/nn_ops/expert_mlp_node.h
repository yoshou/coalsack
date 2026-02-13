#pragma once

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "../dynamic_mx_tensor_message.h"
#include "../nn_op_node.h"

namespace coalsack {

// Expert MLP layer for Mixture-of-Experts (MoE) transformer models
// Implements gated MLP: down(activation(gate(x), up(x)))
// Uses modified SwiGLU activation with value clipping
//
// Architecture:
// - 8 inputs: hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output
// - Weight format: 3D tensors [num_experts, expert_ffn_dim, hidden_dim] in FLOAT16 or MXFP4
// - Bias format: 2D tensors [num_experts, expert_ffn_dim or hidden_dim] in FLOAT32
// - Router output: 4D tensor [batch, seq_len, top_k, 2] where last dim is [expert_id, weight]
class expert_mlp_node : public graph_node {
 public:
  explicit expert_mlp_node(int expert_id = 0)
      : graph_node(), expert_id_(expert_id), output_(std::make_shared<graph_edge>(this)) {
    set_output(output_);
  }

  int get_expert_id() const { return expert_id_; }

  std::string get_proc_name() const override { return "expert_mlp"; }

  void set_input_names(const std::vector<std::string>& names) { input_names_ = names; }
  void set_output_name(const std::string& name) { output_name_ = name; }
  void set_node_name(const std::string& name) { node_name_ = name; }

  std::vector<std::string> get_input_names() const { return input_names_; }
  std::string get_output_name() const { return output_name_; }

  // Public wrapper for testing
  dynamic_tensor compute_test(const std::vector<dynamic_tensor>& inputs) {
    return compute_fp16(inputs);
  }

  // Public wrapper for MXFP4 testing
  void compute_test_mxfp4(const float* hidden_vec, const uint8_t* w_up_data,
                           const uint8_t* w_gate_data, const uint8_t* w_down_data,
                           const float* b_up_data, const float* b_gate_data,
                           const float* b_down_data, float* output_vec, int64_t hidden_dim,
                           int64_t expert_ffn_dim) {
    compute_token_mxfp4(hidden_vec, w_up_data, w_gate_data, w_down_data, b_up_data, b_gate_data,
                         b_down_data, output_vec, hidden_dim, expert_ffn_dim);
  }

  // Override process() to handle both FP16 and MXFP4 weights
  void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    double timestamp = 0.0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
      timestamp = result_msg->get_timestamp();
    }

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping {} [{}] (Frame: {})", "expert_mlp", node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_msg->set_timestamp(timestamp);
      output_->send(output_msg);
      return;
    }
    spdlog::trace("Executing {} [{}] (Frame: {})", "expert_mlp", node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      if (!result_msg || input_names_.empty()) {
        throw std::runtime_error("expert_mlp: invalid input");
      }

      // Get all inputs as graph messages
      std::vector<graph_message_ptr> input_msgs;
      for (const auto& name : input_names_) {
        auto field = result_msg->get_field(name);
        if (!field) {
          throw std::runtime_error("expert_mlp: missing field " + name);
        }
        input_msgs.push_back(field);
      }

      // Check weight message types (indices 1-3: w_up, w_gate, w_down)
      bool is_mxfp4 = false;
      if (auto mx_msg = std::dynamic_pointer_cast<dynamic_mx_tensor_message>(input_msgs[1])) {
        is_mxfp4 = true;
      }

      dynamic_tensor output;
      if (is_mxfp4) {
        // MXFP4 path
        output = compute_with_mxfp4(input_msgs);
      } else {
        // Standard FP16 path
        std::vector<dynamic_tensor> inputs;
        for (const auto& msg : input_msgs) {
          auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(msg);
          if (!tensor_msg) {
            throw std::runtime_error("expert_mlp: expected dynamic_tensor_message");
          }
          inputs.push_back(tensor_msg->get_tensor());
        }
        output = compute_fp16(inputs);
      }

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);
      std::unordered_map<std::string, graph_message_ptr> output_fields;
      output_fields[output_name_] = output_tensor_msg;
      output_msg = result_message::ok(output_fields);
      output_msg->set_frame_number(frame_number);
      output_msg->set_timestamp(timestamp);
    } catch (const std::exception& e) {
      spdlog::error("{} [{}]: {}", "expert_mlp", node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> output_fields;
      output_fields[output_name_] = nullptr;
      output_msg = result_message::error(output_fields, e.what());
      output_msg->set_frame_number(frame_number);
      output_msg->set_timestamp(timestamp);
    }
    output_->send(output_msg);
  }

 private:
  int expert_id_;
  graph_edge_ptr output_;
  std::vector<std::string> input_names_;
  std::string output_name_;
  std::string node_name_;

  // SIMD register structure for 32 FP32 elements (4 x __m256)
  struct simd_block_32 {
    __m256 v0, v1, v2, v3;
  };

  // Compile-time E8M0 to FP32 conversion helper
  static constexpr float compute_e8m0_entry(uint8_t e) {
    if (e < 2) {
      // Denormalized: 2^(-128), 2^(-127)
      uint32_t bits = 0x00200000U << e;
      return std::bit_cast<float>(bits);
    } else {
      // Normalized: 2^(e-128)
      uint32_t bits = static_cast<uint32_t>(e - 1) << 23;
      return std::bit_cast<float>(bits);
    }
  }

  // Generate E8M0 LUT at compile time using index sequence
  template <size_t... Is>
  static constexpr std::array<float, 256> make_e8m0_lut_impl(std::index_sequence<Is...>) {
    return {{compute_e8m0_entry(Is)...}};
  }

  static constexpr std::array<float, 256> make_e8m0_lut() {
    return make_e8m0_lut_impl(std::make_index_sequence<256>{});
  }

  // E8M0 to FP32 lookup (divided by 2, matching e8m0_to_fp32_half)
  static inline float e8m0_to_fp32_half_lut(uint8_t e) {
    static constexpr auto lut = make_e8m0_lut();
    return lut[e];
  }

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

  // SIMD MXFP4 block dequantization (32 elements, fully unrolled)
  // Uses pshufb for 4-bit → int8 table lookup, then converts to float32
  // Returns SIMD registers directly without memory store
  __attribute__((target("avx2"))) static inline simd_block_32 dequantize_mxfp4_block_32(
      const uint8_t* block_data) {
    uint8_t e = block_data[0];
    const uint8_t* qs = block_data + 1;

    // E8M0 to float32 using lookup table
    float scale = e8m0_to_fp32_half_lut(e);
    __m256 scale_vec = _mm256_set1_ps(scale);

    // Load kvalues_mxfp4 lookup table into SSE register
    const __m128i kvalues_vec =
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

    // Load 16 packed bytes
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));

    // Extract lower and upper nibbles
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i qs_lo = _mm_and_si128(packed, lo_mask);
    __m128i qs_hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    // Table lookup using pshufb
    __m128i vals_lo = _mm_shuffle_epi8(kvalues_vec, qs_lo);  // 16 x int8
    __m128i vals_hi = _mm_shuffle_epi8(kvalues_vec, qs_hi);  // 16 x int8

    // Convert and scale, returning SIMD registers directly
    simd_block_32 result;
    result.v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_lo)), scale_vec);
    result.v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_lo, 8))),
                              scale_vec);
    result.v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_hi)), scale_vec);
    result.v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals_hi, 8))),
                              scale_vec);
    return result;
  }

  // Load 32 FP32 elements into SIMD registers
  __attribute__((target("avx2"))) static inline simd_block_32 load_fp32_block_32(
      const float* data) {
    simd_block_32 result;
    result.v0 = _mm256_loadu_ps(data + 0);
    result.v1 = _mm256_loadu_ps(data + 8);
    result.v2 = _mm256_loadu_ps(data + 16);
    result.v3 = _mm256_loadu_ps(data + 24);
    return result;
  }

  // FMA accumulate: acc += hidden.v0*w.v0 + hidden.v1*w.v1 + hidden.v2*w.v2 + hidden.v3*w.v3
  __attribute__((target("avx2,fma"))) static inline void fma_block_32(__m256& acc,
                                                                       const simd_block_32& hidden,
                                                                       const simd_block_32& w) {
    acc = _mm256_fmadd_ps(hidden.v0, w.v0, acc);
    acc = _mm256_fmadd_ps(hidden.v1, w.v1, acc);
    acc = _mm256_fmadd_ps(hidden.v2, w.v2, acc);
    acc = _mm256_fmadd_ps(hidden.v3, w.v3, acc);
  }

  // Compute with FP16 weights
  dynamic_tensor compute_fp16(const std::vector<dynamic_tensor>& inputs) {
    // Validate input count
    if (inputs.size() != 8) {
      throw std::runtime_error(
          "expert_mlp: expected 8 inputs (hidden_states, weights, biases, router_output), got " +
          std::to_string(inputs.size()));
    }

    const auto& hidden_states = inputs[0];
    const auto& w_up = inputs[1];
    const auto& w_gate = inputs[2];
    const auto& w_down = inputs[3];
    const auto& b_up = inputs[4];
    const auto& b_gate = inputs[5];
    const auto& b_down = inputs[6];
    const auto& router_output = inputs[7];

    // Validate router_output shape: [batch, seq_len, top_k, 2]
    if (router_output.ndim() != 4 || router_output.dim(3) != 2) {
      throw std::runtime_error(
          "expert_mlp: router_output must be 4D [batch, seq_len, top_k, 2], got " +
          std::to_string(router_output.ndim()) + "D");
    }
    if (router_output.get_dtype() != dtype::float32) {
      throw std::runtime_error("expert_mlp: router_output must be float32");
    }

    // Check if this expert is selected by any token (early return optimization)
    if (!has_any_token_selecting_expert(router_output, expert_id_)) {
      // Return empty tensor to signal non-selection
      return dynamic_tensor(dtype::float32, {0});
    }

    // Validate hidden_states shape: [batch, seq_len, hidden_dim]
    if (hidden_states.ndim() != 3) {
      throw std::runtime_error(
          "expert_mlp: hidden_states must be 3D [batch, seq_len, hidden_dim], got " +
          std::to_string(hidden_states.ndim()) + "D");
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);

    // Validate weight shapes: all must be 2D (single expert slices)
    // Expected: w_up/w_gate [expert_ffn_dim, hidden_dim], w_down [hidden_dim, expert_ffn_dim]
    if (w_up.ndim() != 2 || w_gate.ndim() != 2 || w_down.ndim() != 2) {
      throw std::runtime_error(
          "expert_mlp: all weights must be 2D [expert_ffn_dim, hidden_dim] or [hidden_dim, "
          "expert_ffn_dim]");
    }

    int64_t expert_ffn_dim = w_up.dim(0);

    // Validate weight dimensions match
    if (w_up.dim(0) != expert_ffn_dim || w_up.dim(1) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w_up must be [" + std::to_string(expert_ffn_dim) +
                               ", " + std::to_string(hidden_dim) + "], got [" +
                               std::to_string(w_up.dim(0)) + ", " + std::to_string(w_up.dim(1)) +
                               "]");
    }
    if (w_gate.dim(0) != expert_ffn_dim || w_gate.dim(1) != hidden_dim) {
      throw std::runtime_error("expert_mlp: w_gate must be [" + std::to_string(expert_ffn_dim) +
                               ", " + std::to_string(hidden_dim) + "], got [" +
                               std::to_string(w_gate.dim(0)) + ", " +
                               std::to_string(w_gate.dim(1)) + "]");
    }
    if (w_down.dim(0) != hidden_dim || w_down.dim(1) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: w_down must be [" + std::to_string(hidden_dim) + ", " +
                               std::to_string(expert_ffn_dim) + "], got [" +
                               std::to_string(w_down.dim(0)) + ", " +
                               std::to_string(w_down.dim(1)) + "]");
    }

    // Validate bias shapes: [expert_ffn_dim] or [hidden_dim] (1D slices)
    if (b_up.ndim() != 1 || b_up.dim(0) != expert_ffn_dim) {
      throw std::runtime_error(
          "expert_mlp: b_up must be [expert_ffn_dim=" + std::to_string(expert_ffn_dim) +
          "], got [" + std::to_string(b_up.dim(0)) + "]");
    }
    if (b_gate.ndim() != 1 || b_gate.dim(0) != expert_ffn_dim) {
      throw std::runtime_error(
          "expert_mlp: b_gate must be [expert_ffn_dim=" + std::to_string(expert_ffn_dim) +
          "], got [" + std::to_string(b_gate.dim(0)) + "]");
    }
    if (b_down.ndim() != 1 || b_down.dim(0) != hidden_dim) {
      throw std::runtime_error(
          "expert_mlp: b_down must be [hidden_dim=" + std::to_string(hidden_dim) + "], got [" +
          std::to_string(b_down.dim(0)) + "]");
    }

    // Validate dtype: weights must be FLOAT16
    if (w_up.get_dtype() != dtype::float16 || w_gate.get_dtype() != dtype::float16 ||
        w_down.get_dtype() != dtype::float16) {
      throw std::runtime_error("expert_mlp: all weights must be FLOAT16");
    }

    // Validate hidden_states dtype
    if (hidden_states.get_dtype() != dtype::float32) {
      throw std::runtime_error("expert_mlp: hidden_states must be FLOAT32");
    }

    // Allocate output: same shape as hidden_states
    dynamic_tensor output(dtype::float32, hidden_states.shape());

    // Compute gated MLP with biases (with per-token conditional execution)
    compute_impl(hidden_states, w_up, w_gate, w_down, b_up, b_gate, b_down, router_output, output,
                 batch, seq_len, hidden_dim, expert_ffn_dim);

    return output;
  }

  // Compute with MXFP4 weights
  dynamic_tensor compute_with_mxfp4(const std::vector<graph_message_ptr>& input_msgs) {
    // Extract inputs
    auto hidden_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(input_msgs[0]);
    auto w_up_msg = std::dynamic_pointer_cast<dynamic_mx_tensor_message>(input_msgs[1]);
    auto w_gate_msg = std::dynamic_pointer_cast<dynamic_mx_tensor_message>(input_msgs[2]);
    auto w_down_msg = std::dynamic_pointer_cast<dynamic_mx_tensor_message>(input_msgs[3]);
    auto b_up_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(input_msgs[4]);
    auto b_gate_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(input_msgs[5]);
    auto b_down_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(input_msgs[6]);
    auto router_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(input_msgs[7]);

    if (!hidden_msg || !w_up_msg || !w_gate_msg || !w_down_msg || !b_up_msg || !b_gate_msg ||
        !b_down_msg || !router_msg) {
      throw std::runtime_error("expert_mlp: invalid input message types for MXFP4");
    }

    const auto& hidden_states = hidden_msg->get_tensor();
    const auto& w_up_mx = w_up_msg->get_mx_tensor();
    const auto& w_gate_mx = w_gate_msg->get_mx_tensor();
    const auto& w_down_mx = w_down_msg->get_mx_tensor();
    const auto& b_up = b_up_msg->get_tensor();
    const auto& b_gate = b_gate_msg->get_tensor();
    const auto& b_down = b_down_msg->get_tensor();
    const auto& router_output = router_msg->get_tensor();

    // Validate shapes
    if (hidden_states.ndim() != 3) {
      throw std::runtime_error("expert_mlp: hidden_states must be 3D");
    }
    if (router_output.ndim() != 4 || router_output.dim(3) != 2) {
      throw std::runtime_error("expert_mlp: router_output must be 4D [batch, seq_len, top_k, 2]");
    }

    // Check if this expert is selected
    if (!has_any_token_selecting_expert(router_output, expert_id_)) {
      return dynamic_tensor(dtype::float32, {0});
    }

    int64_t batch = hidden_states.dim(0);
    int64_t seq_len = hidden_states.dim(1);
    int64_t hidden_dim = hidden_states.dim(2);

    // Get expert_ffn_dim from weight shapes
    if (w_up_mx.ndim() != 2 || w_gate_mx.ndim() != 2 || w_down_mx.ndim() != 2) {
      throw std::runtime_error("expert_mlp: all weights must be 2D");
    }
    int64_t expert_ffn_dim = w_up_mx.dim(0);

    // Validate dimensions
    if (w_up_mx.dim(1) != hidden_dim || w_gate_mx.dim(0) != expert_ffn_dim ||
        w_gate_mx.dim(1) != hidden_dim || w_down_mx.dim(0) != hidden_dim ||
        w_down_mx.dim(1) != expert_ffn_dim) {
      throw std::runtime_error("expert_mlp: weight dimension mismatch");
    }

    // MXFP4 requires dimensions to be multiples of 32
    constexpr int64_t QK_MXFP4 = 32;
    if (hidden_dim % QK_MXFP4 != 0) {
      throw std::runtime_error("expert_mlp: hidden_dim must be a multiple of 32 for MXFP4, got " +
                               std::to_string(hidden_dim));
    }
    if (expert_ffn_dim % QK_MXFP4 != 0) {
      throw std::runtime_error(
          "expert_mlp: expert_ffn_dim must be a multiple of 32 for MXFP4, got " +
          std::to_string(expert_ffn_dim));
    }

    // Allocate output
    dynamic_tensor output(dtype::float32, hidden_states.shape());

    // Compute
    compute_impl_mxfp4(hidden_states, w_up_mx, w_gate_mx, w_down_mx, b_up, b_gate, b_down,
                       router_output, output, batch, seq_len, hidden_dim, expert_ffn_dim);

    return output;
  }

  // Compute implementation for MXFP4 weights
  void compute_impl_mxfp4(const dynamic_tensor& hidden_states, const dynamic_mx_tensor& w_up_mx,
                          const dynamic_mx_tensor& w_gate_mx, const dynamic_mx_tensor& w_down_mx,
                          const dynamic_tensor& b_up, const dynamic_tensor& b_gate,
                          const dynamic_tensor& b_down, const dynamic_tensor& router_output,
                          dynamic_tensor& output, int64_t batch, int64_t seq_len,
                          int64_t hidden_dim, int64_t expert_ffn_dim) {
    const float* hidden_data = hidden_states.data_ptr<float>();
    const uint8_t* w_up_data = w_up_mx.data_ptr();
    const uint8_t* w_gate_data = w_gate_mx.data_ptr();
    const uint8_t* w_down_data = w_down_mx.data_ptr();
    const float* b_up_data = b_up.data_ptr<float>();
    const float* b_gate_data = b_gate.data_ptr<float>();
    const float* b_down_data = b_down.data_ptr<float>();
    float* out_data = output.data_ptr<float>();
    const float* router_data = router_output.data_ptr<float>();
    int64_t top_k = router_output.dim(2);

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const float* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        float* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        // Check if this token selected current expert_id_
        if (!is_token_selecting_expert(router_data, b, s, seq_len, top_k, expert_id_)) {
          std::memset(output_vec, 0, hidden_dim * sizeof(float));
          continue;
        }

        // Process selected token with MXFP4
        compute_token_mxfp4(hidden_vec, w_up_data, w_gate_data, w_down_data, b_up_data, b_gate_data,
                            b_down_data, output_vec, hidden_dim, expert_ffn_dim);
      }
    }
  }

  // Convert fp16 → fp32 using F16C + AVX2 SIMD
  __attribute__((target("f16c,avx2,fma"))) void compute_token(
      const float* hidden_vec, const uint16_t* w_up_data, const uint16_t* w_gate_data,
      const uint16_t* w_down_data, const float* b_up_data, const float* b_gate_data,
      const float* b_down_data, float* output_vec, int64_t hidden_dim,
      int64_t expert_ffn_dim) const {
    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;

    // Allocate intermediate buffer for activated values
    std::vector<float> activated(expert_ffn_dim);

    // Step 1: Up/gate projections with biases, then modified SwiGLU activation
    // Weights: [expert_ffn_dim, hidden_dim]
    // Biases: [expert_ffn_dim]
    // Access w[out_idx][in_idx] via: out_idx * hidden_dim + in_idx
    // Access b[out_idx] via: out_idx
    for (int64_t out_idx = 0; out_idx < expert_ffn_dim; ++out_idx) {
      __m256 up_vec = _mm256_setzero_ps();
      __m256 gate_vec = _mm256_setzero_ps();

      int64_t in_idx = 0;
      // Process 8 elements at a time with AVX2
      for (; in_idx + 7 < hidden_dim; in_idx += 8) {
        // Weight index: w[out_idx][in_idx]
        int64_t weight_idx = out_idx * hidden_dim + in_idx;

        // Load 8 FP16 weights and convert to FP32
        __m128i w_up_fp16 = _mm_loadu_si128((__m128i*)(w_up_data + weight_idx));
        __m128i w_gate_fp16 = _mm_loadu_si128((__m128i*)(w_gate_data + weight_idx));
        __m256 w_up_fp32 = _mm256_cvtph_ps(w_up_fp16);
        __m256 w_gate_fp32 = _mm256_cvtph_ps(w_gate_fp16);

        // Load 8 hidden states
        __m256 hidden = _mm256_loadu_ps(hidden_vec + in_idx);

        // FMA: up_vec += hidden * w_up, gate_vec += hidden * w_gate
        up_vec = _mm256_fmadd_ps(hidden, w_up_fp32, up_vec);
        gate_vec = _mm256_fmadd_ps(hidden, w_gate_fp32, gate_vec);
      }

      // Horizontal sum using SIMD intrinsics
      float up_sum = horizontal_sum_avx2(up_vec);
      float gate_sum = horizontal_sum_avx2(gate_vec);

      // Process remaining elements
      for (; in_idx < hidden_dim; ++in_idx) {
        // Weight index: w[out_idx][in_idx]
        int64_t weight_idx = out_idx * hidden_dim + in_idx;
        float w_up_f32 = _cvtsh_ss(w_up_data[weight_idx]);
        float w_gate_f32 = _cvtsh_ss(w_gate_data[weight_idx]);
        up_sum += hidden_vec[in_idx] * w_up_f32;
        gate_sum += hidden_vec[in_idx] * w_gate_f32;
      }

      // Add biases: b[out_idx]
      const float up_v = up_sum + b_up_data[out_idx];
      const float gate_v = gate_sum + b_gate_data[out_idx];

      // Modified SwiGLU activation with clipping:
      //   gate_clipped = min(gate_v, limit)
      //   up_clipped = clamp(up_v, -limit, limit)
      //   activation = (gate_clipped / (1 + exp(-alpha * gate_clipped))) * (up_clipped + 1)
      const float x = std::min(gate_v, limit);
      const float y = std::clamp(up_v, -limit, limit);
      const float out_glu = x / (1.0f + std::exp(alpha * (-x)));

      activated[out_idx] = out_glu * (y + 1.0f);
    }

    // Step 2: Down projection with bias
    // Weights: [hidden_dim, expert_ffn_dim]
    // Biases: [hidden_dim]
    // Access w[out_idx][in_idx] via: out_idx * expert_ffn_dim + in_idx
    // Access b[out_idx] via: out_idx
    for (int64_t out_idx = 0; out_idx < hidden_dim; ++out_idx) {
      __m256 sum_vec = _mm256_setzero_ps();

      int64_t in_idx = 0;
      // Process 8 elements at a time with AVX2
      for (; in_idx + 7 < expert_ffn_dim; in_idx += 8) {
        // Weight index: w_down[out_idx][in_idx]
        int64_t weight_idx = out_idx * expert_ffn_dim + in_idx;

        // Load 8 FP16 weights and convert to FP32
        __m128i w_down_fp16 = _mm_loadu_si128((__m128i*)(w_down_data + weight_idx));
        __m256 w_down_fp32 = _mm256_cvtph_ps(w_down_fp16);

        // Load 8 activated values
        __m256 act = _mm256_loadu_ps(activated.data() + in_idx);

        // FMA: sum_vec += act * w_down
        sum_vec = _mm256_fmadd_ps(act, w_down_fp32, sum_vec);
      }

      // Horizontal sum using SIMD intrinsics
      float sum = horizontal_sum_avx2(sum_vec);

      // Process remaining elements
      for (; in_idx < expert_ffn_dim; ++in_idx) {
        // Weight index: w_down[out_idx][in_idx]
        int64_t weight_idx = out_idx * expert_ffn_dim + in_idx;
        float w_down_f32 = _cvtsh_ss(w_down_data[weight_idx]);
        sum += activated[in_idx] * w_down_f32;
      }

      // Add down projection bias: b[out_idx]
      output_vec[out_idx] = sum + b_down_data[out_idx];
    }
  }

  // MXFP4 on-the-fly dequantization and computation with SIMD
  // Uses accumulator optimization (single horizontal sum per output) + FMA + output tiling N=2
  __attribute__((target("f16c,avx2,fma"))) void compute_token_mxfp4(
      const float* hidden_vec, const uint8_t* w_up_data, const uint8_t* w_gate_data,
      const uint8_t* w_down_data, const float* b_up_data, const float* b_gate_data,
      const float* b_down_data, float* output_vec, int64_t hidden_dim,
      int64_t expert_ffn_dim) const {
    constexpr int TILE = 2;
    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;
    constexpr size_t QK_MXFP4 = 32;
    constexpr size_t MXFP4_BLOCK_BYTES = 17;

    if (expert_ffn_dim % TILE != 0) {
      throw std::runtime_error(
          "compute_token_mxfp4: expert_ffn_dim must be even, got " +
          std::to_string(expert_ffn_dim));
    }
    if (hidden_dim % TILE != 0) {
      throw std::runtime_error(
          "compute_token_mxfp4: hidden_dim must be even, got " +
          std::to_string(hidden_dim));
    }

    std::vector<float> activated(expert_ffn_dim);

    int64_t num_blocks_hidden = hidden_dim / QK_MXFP4;
    int64_t num_blocks_ffn = expert_ffn_dim / QK_MXFP4;

    // Step 1: Up/gate projections with output tiling N=2
    for (int64_t out_idx = 0; out_idx < expert_ffn_dim; out_idx += TILE) {
      __m256 up_acc[TILE], gate_acc[TILE];
      for (int t = 0; t < TILE; ++t) {
        up_acc[t] = _mm256_setzero_ps();
        gate_acc[t] = _mm256_setzero_ps();
      }

      for (int64_t blk = 0; blk < num_blocks_hidden; ++blk) {
        auto hidden_simd = load_fp32_block_32(hidden_vec + blk * QK_MXFP4);

        for (int t = 0; t < TILE; ++t) {
          int64_t offset = ((out_idx + t) * num_blocks_hidden + blk) * MXFP4_BLOCK_BYTES;
          fma_block_32(up_acc[t], hidden_simd, dequantize_mxfp4_block_32(w_up_data + offset));
          fma_block_32(gate_acc[t], hidden_simd, dequantize_mxfp4_block_32(w_gate_data + offset));
        }
      }

      for (int t = 0; t < TILE; ++t) {
        const float up_v = horizontal_sum_avx2(up_acc[t]) + b_up_data[out_idx + t];
        const float gate_v = horizontal_sum_avx2(gate_acc[t]) + b_gate_data[out_idx + t];
        const float x = std::min(gate_v, limit);
        const float y = std::clamp(up_v, -limit, limit);
        const float out_glu = x / (1.0f + std::exp(alpha * (-x)));
        activated[out_idx + t] = out_glu * (y + 1.0f);
      }
    }

    // Step 2: Down projection with output tiling N=2
    for (int64_t out_idx = 0; out_idx < hidden_dim; out_idx += TILE) {
      __m256 acc[TILE];
      for (int t = 0; t < TILE; ++t) {
        acc[t] = _mm256_setzero_ps();
      }

      for (int64_t blk = 0; blk < num_blocks_ffn; ++blk) {
        auto act_simd = load_fp32_block_32(activated.data() + blk * QK_MXFP4);

        for (int t = 0; t < TILE; ++t) {
          int64_t offset = ((out_idx + t) * num_blocks_ffn + blk) * MXFP4_BLOCK_BYTES;
          fma_block_32(acc[t], act_simd, dequantize_mxfp4_block_32(w_down_data + offset));
        }
      }

      for (int t = 0; t < TILE; ++t) {
        output_vec[out_idx + t] = horizontal_sum_avx2(acc[t]) + b_down_data[out_idx + t];
      }
    }
  }

  // Modified SwiGLU activation with clipping:
  //   gate_clipped = min(gate, limit)
  //   up_clipped = clamp(up, -limit, limit)
  //   activation = (gate_clipped / (1 + exp(alpha * (-gate_clipped)))) * (up_clipped + 1)
  // Followed by down projection: down(activation) + b_down
  // Token-level conditional execution: only compute for tokens that selected this expert
  void compute_impl(const dynamic_tensor& hidden_states, const dynamic_tensor& w_up,
                    const dynamic_tensor& w_gate, const dynamic_tensor& w_down,
                    const dynamic_tensor& b_up, const dynamic_tensor& b_gate,
                    const dynamic_tensor& b_down, const dynamic_tensor& router_output,
                    dynamic_tensor& output, int64_t batch, int64_t seq_len, int64_t hidden_dim,
                    int64_t expert_ffn_dim) {
    const float* hidden_data = hidden_states.data_ptr<float>();
    const uint16_t* w_up_data = w_up.data_ptr<uint16_t>();
    const uint16_t* w_gate_data = w_gate.data_ptr<uint16_t>();
    const uint16_t* w_down_data = w_down.data_ptr<uint16_t>();
    const float* b_up_data = b_up.data_ptr<float>();
    const float* b_gate_data = b_gate.data_ptr<float>();
    const float* b_down_data = b_down.data_ptr<float>();
    float* out_data = output.data_ptr<float>();
    const float* router_data = router_output.data_ptr<float>();
    int64_t top_k = router_output.dim(2);

    // For each token position
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        const float* hidden_vec = hidden_data + (b * seq_len + s) * hidden_dim;
        float* output_vec = out_data + (b * seq_len + s) * hidden_dim;

        // Check if this token selected current expert_id_
        if (!is_token_selecting_expert(router_data, b, s, seq_len, top_k, expert_id_)) {
          // Zero out output for non-selected tokens
          std::memset(output_vec, 0, hidden_dim * sizeof(float));
          continue;
        }

        // Process selected token
        compute_token(hidden_vec, w_up_data, w_gate_data, w_down_data, b_up_data, b_gate_data,
                      b_down_data, output_vec, hidden_dim, expert_ffn_dim);
      }
    }
  }

  // Check if expert_id is in selected_expert_ids (O(top_k) linear search, typically ~4 elements)
  // Check if a specific token selected this expert
  bool is_token_selecting_expert(const float* router_data, int64_t b, int64_t s, int64_t seq_len,
                                 int64_t top_k, int expert_id) const {
    int64_t base_idx = (b * seq_len + s) * top_k * 2;
    for (int64_t k = 0; k < top_k; ++k) {
      if (static_cast<int>(router_data[base_idx + k * 2]) == expert_id) {
        return true;
      }
    }
    return false;
  }

  // Check if any token in the batch selected this expert (for early return)
  bool has_any_token_selecting_expert(const dynamic_tensor& router_output, int expert_id) const {
    const float* router_data = router_output.data_ptr<float>();
    int64_t batch = router_output.dim(0);
    int64_t seq_len = router_output.dim(1);
    int64_t top_k = router_output.dim(2);

    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        if (is_token_selecting_expert(router_data, b, s, seq_len, top_k, expert_id)) {
          return true;
        }
      }
    }
    return false;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::expert_mlp_node, coalsack::graph_node)
