#pragma once

#include <limits>
#include <numeric>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class max_pool_node : public unary_op_node {
 public:
  max_pool_node()
      : unary_op_node("max_pool"), kernel_shape_({2, 2}), strides_({1, 1}), pads_({0, 0, 0, 0}) {}

  void set_kernel_shape(const std::vector<int64_t>& kernel_shape) { kernel_shape_ = kernel_shape; }
  void set_strides(const std::vector<int64_t>& strides) { strides_ = strides; }
  void set_pads(const std::vector<int64_t>& pads) { pads_ = pads; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& shape = input.shape();
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t kH = kernel_shape_[0], kW = kernel_shape_[1];
    int64_t H_out = (H + pads_[0] + pads_[2] - kH) / strides_[0] + 1;
    int64_t W_out = (W + pads_[1] + pads_[3] - kW) / strides_[1] + 1;

    dynamic_tensor output(input.get_dtype(), {N, C, H_out, W_out});

    if (input.get_dtype() == dtype::float32) {
      max_pool_impl<float>(input, output, N, C, H, W, H_out, W_out, kH, kW);
    } else if (input.get_dtype() == dtype::float64) {
      max_pool_impl<double>(input, output, N, C, H, W, H_out, W_out, kH, kW);
    } else {
      throw std::runtime_error("max_pool: unsupported dtype");
    }

    return output;
  }

 private:
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;

  template <typename T>
  void max_pool_impl(const dynamic_tensor& input, dynamic_tensor& output, int64_t N, int64_t C,
                     int64_t H, int64_t W, int64_t H_out, int64_t W_out, int64_t kH, int64_t kW) {
    const T* in_data = input.data_ptr<T>();
    T* out_data = output.data_ptr<T>();

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t h_out = 0; h_out < H_out; ++h_out) {
          for (int64_t w_out = 0; w_out < W_out; ++w_out) {
            T max_val = -std::numeric_limits<T>::infinity();
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                int64_t h_in = h_out * strides_[0] - pads_[0] + kh;
                int64_t w_in = w_out * strides_[1] - pads_[1] + kw;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                  max_val = std::max(max_val, in_data[n * C * H * W + c * H * W + h_in * W + w_in]);
                }
              }
            }
            out_data[n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out] = max_val;
          }
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::max_pool_node, coalsack::graph_node)
