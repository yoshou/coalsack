#pragma once

#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class grid_sample_node : public binary_op_node {
 public:
  grid_sample_node()
      : binary_op_node("grid_sample"),
        mode_("bilinear"),
        padding_mode_("zeros"),
        align_corners_(0) {}

  void set_mode(const std::string& mode) { mode_ = mode; }
  void set_padding_mode(const std::string& padding_mode) { padding_mode_ = padding_mode; }
  void set_align_corners(int64_t align_corners) { align_corners_ = align_corners; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input, const dynamic_tensor& grid) override {
    if (input.get_dtype() != dtype::float32) {
      throw std::runtime_error("grid_sample: only float32 supported");
    }

    const auto& in_shape = input.shape();
    const auto& grid_shape = grid.shape();

    std::vector<int64_t> out_shape = {in_shape[0], in_shape[1], grid_shape[1], grid_shape[2]};
    dynamic_tensor output(input.get_dtype(), out_shape);

    grid_sample_impl(input, grid, output);

    return output;
  }

 private:
  std::string mode_;
  std::string padding_mode_;
  int64_t align_corners_;

  void grid_sample_impl(const dynamic_tensor& input, const dynamic_tensor& grid,
                        dynamic_tensor& output) {
    const float* in_data = input.data_ptr<float>();
    const float* grid_data = grid.data_ptr<float>();
    float* out_data = output.data_ptr<float>();

    const auto& in_shape = input.shape();
    int64_t N = in_shape[0], C = in_shape[1], H_in = in_shape[2], W_in = in_shape[3];
    int64_t H_out = grid.shape()[1], W_out = grid.shape()[2];

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H_out; ++h) {
          for (int64_t w = 0; w < W_out; ++w) {
            int64_t grid_idx = n * H_out * W_out * 2 + h * W_out * 2 + w * 2;
            float x = grid_data[grid_idx];
            float y = grid_data[grid_idx + 1];

            float px, py;
            if (align_corners_) {
              px = ((x + 1) / 2) * (W_in - 1);
              py = ((y + 1) / 2) * (H_in - 1);
            } else {
              px = ((x + 1) * W_in - 1) / 2.0f;
              py = ((y + 1) * H_in - 1) / 2.0f;
            }

            int64_t x0_idx = static_cast<int64_t>(std::floor(px));
            int64_t y0_idx = static_cast<int64_t>(std::floor(py));
            int64_t x1_idx = x0_idx + 1;
            int64_t y1_idx = y0_idx + 1;

            auto get_val = [&](int64_t xx, int64_t yy) {
              if (xx < 0 || xx >= W_in || yy < 0 || yy >= H_in) return 0.0f;
              return in_data[n * C * H_in * W_in + c * H_in * W_in + yy * W_in + xx];
            };

            float v00 = get_val(x0_idx, y0_idx);
            float v01 = get_val(x1_idx, y0_idx);
            float v10 = get_val(x0_idx, y1_idx);
            float v11 = get_val(x1_idx, y1_idx);

            float wx1 = px - x0_idx;
            float wy1 = py - y0_idx;
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;

            float val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

            int64_t out_idx = n * C * H_out * W_out + c * H_out * W_out + h * W_out + w;
            out_data[out_idx] = val;
          }
        }
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::grid_sample_node, coalsack::graph_node)
