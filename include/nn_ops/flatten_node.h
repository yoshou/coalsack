#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class flatten_node : public unary_op_node {
 public:
  flatten_node() : unary_op_node("flatten"), axis_(1) {}

  void set_axis(int64_t axis) { axis_ = axis; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    const auto& shape = input.shape();
    int64_t axis = axis_ < 0 ? shape.size() + axis_ : axis_;

    if (axis < 0 || axis > static_cast<int64_t>(shape.size())) {
      throw std::runtime_error("flatten: axis out of range");
    }

    int64_t dim0 = 1;
    for (int64_t i = 0; i < axis; ++i) {
      dim0 *= shape[i];
    }

    int64_t dim1 = 1;
    for (int64_t i = axis; i < static_cast<int64_t>(shape.size()); ++i) {
      dim1 *= shape[i];
    }

    dynamic_tensor output(input.get_dtype(), {dim0, dim1});
    std::memcpy(output.data(), input.data(), input.bytes());
    return output;
  }

 private:
  int64_t axis_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::flatten_node, coalsack::graph_node)
