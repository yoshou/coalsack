#pragma once

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class shape_node : public unary_op_node {
 private:
  int64_t start_;
  int64_t end_;

 public:
  shape_node() : unary_op_node("shape"), start_(0), end_(INT64_MAX) {}

  void set_start(int64_t start) { start_ = start; }
  void set_end(int64_t end) { end_ = end; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& shape = input.shape();
    int64_t rank = static_cast<int64_t>(shape.size());

    // Normalize start and end
    int64_t start = start_;
    int64_t end = (end_ == INT64_MAX) ? rank : end_;

    if (start < 0) start += rank;
    if (end < 0) end += rank;

    start = std::max(int64_t(0), std::min(start, rank));
    end = std::max(int64_t(0), std::min(end, rank));

    if (start > end) start = end;

    int64_t output_size = end - start;
    std::vector<int64_t> output_shape = {output_size};
    dynamic_tensor output(dtype::int64, output_shape);

    int64_t* output_data = output.data_ptr<int64_t>();
    for (int64_t i = 0; i < output_size; ++i) {
      output_data[i] = shape[start + i];
    }

    return output;
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::shape_node, coalsack::graph_node)
