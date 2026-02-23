#pragma once

#include <cmath>
#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// 2D Rotary Position Embedding (RoPE) for ViT patch tokens.
// Input/output: [n_pos, n_embd] (n_embd = n_head * d_head)
// Each head is split into two halves: the first half applies RoPE using the column position
// of the patch (pos_w), and the second half uses the row position (pos_h).
class rope_2d_node : public unary_op_node {
 public:
  rope_2d_node() : unary_op_node("rope_2d"), theta_(10000.0f), n_head_(0), d_head_(0) {}

  // Called once per build_graph to set the fixed position ids.
  void set_positions(const std::vector<int>& pos_w, const std::vector<int>& pos_h) {
    pos_w_ = pos_w;
    pos_h_ = pos_h;
  }

  void set_theta(float theta) { theta_ = theta; }
  void set_num_heads(int n_head) { n_head_ = n_head; }
  void set_head_dim(int d_head) { d_head_ = d_head; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() != dtype::float32) {
      throw std::runtime_error("rope_2d: input must be float32");
    }
    if (input.ndim() != 2) {
      throw std::runtime_error("rope_2d: expected 2D input [n_pos, n_embd]");
    }

    int64_t n_pos = input.dim(0);
    int64_t n_embd = input.dim(1);

    if (n_head_ <= 0 || d_head_ <= 0) {
      throw std::runtime_error("rope_2d: n_head and d_head must be set");
    }
    if (n_embd != static_cast<int64_t>(n_head_ * d_head_)) {
      throw std::runtime_error("rope_2d: n_embd != n_head * d_head");
    }
    if (static_cast<int64_t>(pos_w_.size()) < n_pos ||
        static_cast<int64_t>(pos_h_.size()) < n_pos) {
      throw std::runtime_error("rope_2d: position vectors too short for n_pos");
    }

    dynamic_tensor output = input.clone();
    float* data = output.data_ptr<float>();
    const int half = d_head_ / 2;

    for (int64_t p = 0; p < n_pos; ++p) {
      for (int h = 0; h < n_head_; ++h) {
        float* vec = data + p * n_embd + h * d_head_;

        // First half: use pos_w (column position)
        {
          float pos = static_cast<float>(pos_w_[p]);
          for (int i = 0; i < half / 2; ++i) {
            float freq = std::pow(theta_, -2.0f * i / half);
            float angle = pos * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float x = vec[2 * i];
            float y = vec[2 * i + 1];
            vec[2 * i] = x * c - y * s;
            vec[2 * i + 1] = x * s + y * c;
          }
        }

        // Second half: use pos_h (row position)
        {
          float pos = static_cast<float>(pos_h_[p]);
          float* vec2 = vec + half;
          for (int i = 0; i < half / 2; ++i) {
            float freq = std::pow(theta_, -2.0f * i / half);
            float angle = pos * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float x = vec2[2 * i];
            float y = vec2[2 * i + 1];
            vec2[2 * i] = x * c - y * s;
            vec2[2 * i + 1] = x * s + y * c;
          }
        }
      }
    }

    return output;
  }

 private:
  std::vector<int> pos_w_;
  std::vector<int> pos_h_;
  float theta_;
  int n_head_;
  int d_head_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::rope_2d_node, coalsack::graph_node)
