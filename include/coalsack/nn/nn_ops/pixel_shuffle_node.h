#pragma once

#include <cstring>
#include <stdexcept>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// Pixel shuffle for ViT patch tokens.
// Input: [n_pos, n_embd] — the last row (CLS token) is dropped before shuffling.
// Output: [n_merged, merged_embd] where n_merged = (patches_per_row/S)^2, merged_embd = n_embd*S*S.
// Token ordering: column-major (merged_idx = mx * mh + my).
class pixel_shuffle_node : public unary_op_node {
 public:
  pixel_shuffle_node() : unary_op_node("pixel_shuffle"), scale_factor_(2), patches_per_row_(0) {}

  void set_scale_factor(int scale) { scale_factor_ = scale; }
  void set_patches_per_row(int n) { patches_per_row_ = n; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() != dtype::float32) {
      throw std::runtime_error("pixel_shuffle: input must be float32");
    }
    if (input.ndim() != 2) {
      throw std::runtime_error("pixel_shuffle: expected 2D input [n_pos, n_embd]");
    }
    if (patches_per_row_ <= 0) {
      throw std::runtime_error("pixel_shuffle: patches_per_row must be set");
    }

    int64_t n_pos = input.dim(0);
    int64_t n_embd = input.dim(1);
    int64_t n_patches = n_pos - 1;  // drop CLS (last row)

    const int pw = patches_per_row_;
    const int S = scale_factor_;
    const int mw = pw / S;
    const int mh = pw / S;

    if (static_cast<int64_t>(pw * pw) != n_patches) {
      throw std::runtime_error("pixel_shuffle: patches_per_row^2 != n_pos - 1");
    }

    int64_t n_merged = static_cast<int64_t>(mw * mh);
    int64_t merged_embd = n_embd * S * S;

    dynamic_tensor output(dtype::float32, {n_merged, merged_embd});
    float* dst = output.data_ptr<float>();
    const float* src = input.data_ptr<float>();

    for (int my = 0; my < mh; ++my) {
      for (int mx = 0; mx < mw; ++mx) {
        // merged_idx = mx * mh + my (column-major: output tokens are ordered column by column)
        int64_t merged_idx = static_cast<int64_t>(mx) * mh + my;
        float* out_row = dst + merged_idx * merged_embd;

        for (int dy = 0; dy < S; ++dy) {
          for (int dx = 0; dx < S; ++dx) {
            int py = my * S + dy;
            int px = mx * S + dx;
            int64_t src_idx = static_cast<int64_t>(py) * pw + px;
            const float* src_row = src + src_idx * n_embd;
            float* dst_block = out_row + (dy * S + dx) * n_embd;
            std::memcpy(dst_block, src_row, static_cast<size_t>(n_embd) * sizeof(float));
          }
        }
      }
    }

    return output;
  }

 private:
  int scale_factor_;
  int patches_per_row_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::pixel_shuffle_node, coalsack::graph_node)
