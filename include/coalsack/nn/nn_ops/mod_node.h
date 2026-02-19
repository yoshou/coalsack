#pragma once

#include <cmath>

#include "coalsack/nn/nn_op_node.h"
#include "coalsack/nn/nn_ops/elementwise_helpers.h"

namespace coalsack {

class mod_node : public binary_op_node {
 public:
  mod_node() : binary_op_node("mod"), fmod_(0) {}

  void set_fmod(int64_t fmod) { fmod_ = fmod; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.get_dtype() == dtype::float32) {
      if (fmod_ == 0) {
        return elementwise_binary_op(
            a, b, dtype::float32,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              float x = a.data_ptr<float>()[a_idx];
              float y = b.data_ptr<float>()[b_idx];
              float val = (y == 0) ? 0.0f : x - y * std::floor(x / y);
              out.data_ptr<float>()[out_idx] = val;
            });
      } else {
        return elementwise_binary_op(
            a, b, dtype::float32,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              out.data_ptr<float>()[out_idx] =
                  std::fmod(a.data_ptr<float>()[a_idx], b.data_ptr<float>()[b_idx]);
            });
      }
    } else if (a.get_dtype() == dtype::float64) {
      if (fmod_ == 0) {
        return elementwise_binary_op(
            a, b, dtype::float64,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              double x = a.data_ptr<double>()[a_idx];
              double y = b.data_ptr<double>()[b_idx];
              double val = (y == 0) ? 0.0 : x - y * std::floor(x / y);
              out.data_ptr<double>()[out_idx] = val;
            });
      } else {
        return elementwise_binary_op(
            a, b, dtype::float64,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              out.data_ptr<double>()[out_idx] =
                  std::fmod(a.data_ptr<double>()[a_idx], b.data_ptr<double>()[b_idx]);
            });
      }
    } else if (a.get_dtype() == dtype::int32) {
      if (fmod_ == 0) {
        return elementwise_binary_op(
            a, b, dtype::int32,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              int32_t x = a.data_ptr<int32_t>()[a_idx];
              int32_t y = b.data_ptr<int32_t>()[b_idx];
              if (y == 0) {
                out.data_ptr<int32_t>()[out_idx] = 0;
              } else {
                int32_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) {
                  r += y;
                }
                out.data_ptr<int32_t>()[out_idx] = r;
              }
            });
      } else {
        return elementwise_binary_op(
            a, b, dtype::int32,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              int32_t x = a.data_ptr<int32_t>()[a_idx];
              int32_t y = b.data_ptr<int32_t>()[b_idx];
              out.data_ptr<int32_t>()[out_idx] = (y != 0 ? x % y : 0);
            });
      }
    } else if (a.get_dtype() == dtype::int64) {
      if (fmod_ == 0) {
        return elementwise_binary_op(
            a, b, dtype::int64,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              int64_t x = a.data_ptr<int64_t>()[a_idx];
              int64_t y = b.data_ptr<int64_t>()[b_idx];
              if (y == 0) {
                out.data_ptr<int64_t>()[out_idx] = 0;
              } else {
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) {
                  r += y;
                }
                out.data_ptr<int64_t>()[out_idx] = r;
              }
            });
      } else {
        return elementwise_binary_op(
            a, b, dtype::int64,
            [](const dynamic_tensor& a, const dynamic_tensor& b, dynamic_tensor& out, int64_t a_idx,
               int64_t b_idx, int64_t out_idx) {
              int64_t x = a.data_ptr<int64_t>()[a_idx];
              int64_t y = b.data_ptr<int64_t>()[b_idx];
              out.data_ptr<int64_t>()[out_idx] = (y != 0 ? x % y : 0);
            });
      }
    } else {
      throw std::runtime_error("mod: unsupported dtype");
    }
  }

 private:
  int64_t fmod_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::mod_node, coalsack::graph_node)
