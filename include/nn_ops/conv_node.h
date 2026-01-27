#pragma once

#include <algorithm>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class conv_node : public variadic_op_node {
 public:
  conv_node()
      : variadic_op_node("conv", 3),
        strides_({1, 1}),
        pads_({0, 0, 0, 0}),
        dilations_({1, 1}),
        group_(1) {}

  void set_strides(const std::vector<int64_t>& strides) { strides_ = strides; }
  void set_pads(const std::vector<int64_t>& pads) { pads_ = pads; }
  void set_dilations(const std::vector<int64_t>& dilations) { dilations_ = dilations; }
  void set_group(int64_t group) { group_ = group; }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    const auto& X = inputs[0];
    const auto& W = inputs[1];

    if (X.get_dtype() != dtype::float32 || W.get_dtype() != dtype::float32) {
      throw std::runtime_error("conv: only float32 supported");
    }
    if (inputs.size() > 2 && inputs[2].get_dtype() != dtype::float32) {
      throw std::runtime_error("conv: bias must be float32");
    }

    const auto& x_shape = X.shape();
    const auto& w_shape = W.shape();

    int64_t N = x_shape[0], C_in = x_shape[1], H = x_shape[2], W_in = x_shape[3];
    int64_t C_out = w_shape[0], kH = w_shape[2], kW = w_shape[3];

    int64_t H_out = (H + pads_[0] + pads_[2] - dilations_[0] * (kH - 1) - 1) / strides_[0] + 1;
    int64_t W_out = (W_in + pads_[1] + pads_[3] - dilations_[1] * (kW - 1) - 1) / strides_[1] + 1;

    dynamic_tensor output(X.get_dtype(), {N, C_out, H_out, W_out});

    const float* x_data = X.data_ptr<float>();
    const float* w_data = W.data_ptr<float>();
    float* out_data = output.data_ptr<float>();
    std::fill_n(out_data, output.numel(), 0.0f);

    int64_t C_in_per_group = C_in / group_;
    int64_t C_out_per_group = C_out / group_;

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t g = 0; g < static_cast<size_t>(group_); ++g) {
        for (int64_t c_out_offset = 0; c_out_offset < C_out_per_group; ++c_out_offset) {
          int64_t c_out = g * C_out_per_group + c_out_offset;
          for (int64_t h_out = 0; h_out < H_out; ++h_out) {
            for (int64_t w_out = 0; w_out < W_out; ++w_out) {
              double sum = 0.0;
              for (int64_t c_in_offset = 0; c_in_offset < C_in_per_group; ++c_in_offset) {
                int64_t c_in = g * C_in_per_group + c_in_offset;
                for (int64_t kh = 0; kh < kH; ++kh) {
                  for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t h_in = h_out * strides_[0] - pads_[0] + kh * dilations_[0];
                    int64_t w_in = w_out * strides_[1] - pads_[1] + kw * dilations_[1];
                    if (h_in >= 0 && h_in < static_cast<int64_t>(H) && w_in >= 0 &&
                        w_in < static_cast<int64_t>(W_in)) {
                      sum +=
                          (double)
                              x_data[n * C_in * H * W_in + c_in * H * W_in + h_in * W_in + w_in] *
                          (double)w_data[c_out * C_in_per_group * kH * kW + c_in_offset * kH * kW +
                                         kh * kW + kw];
                    }
                  }
                }
              }
              out_data[n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out] =
                  (float)sum;
            }
          }
        }
      }
    }

    if (inputs.size() > 2) {
      const auto& B = inputs[2];
      const float* b_data = B.data_ptr<float>();
      for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
          for (int64_t h = 0; h < H_out; ++h) {
            for (int64_t w = 0; w < W_out; ++w) {
              out_data[n * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w] += b_data[c];
            }
          }
        }
      }
    }

    return output;
  }

 private:
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> dilations_;
  int64_t group_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::conv_node, coalsack::graph_node)
