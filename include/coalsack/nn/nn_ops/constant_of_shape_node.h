#pragma once

#include <algorithm>
#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

class constant_of_shape_node : public unary_op_node {
 public:
  constant_of_shape_node() : unary_op_node("constant_of_shape") {
    value_ = dynamic_tensor(dtype::float32, std::vector<int64_t>{1});
    *value_.data_ptr<float>() = 0.0f;
  }

  void set_value(const dynamic_tensor& value) { value_ = value; }

 protected:
  dynamic_tensor compute(const dynamic_tensor& shape_tensor) override {
    const int64_t* shape_data = shape_tensor.data_ptr<int64_t>();
    std::vector<int64_t> output_shape(shape_data, shape_data + shape_tensor.numel());

    dynamic_tensor output(value_.get_dtype(), output_shape);

    size_t count = output.numel();

    switch (value_.get_dtype()) {
      case dtype::float32: {
        float val = *value_.data_ptr<float>();
        std::fill_n(output.data_ptr<float>(), count, val);
        break;
      }
      case dtype::int64: {
        int64_t val = *value_.data_ptr<int64_t>();
        std::fill_n(output.data_ptr<int64_t>(), count, val);
        break;
      }
      case dtype::int32: {
        int32_t val = *value_.data_ptr<int32_t>();
        std::fill_n(output.data_ptr<int32_t>(), count, val);
        break;
      }
      case dtype::bool_: {
        bool val = *value_.data_ptr<bool>();
        std::fill_n(output.data_ptr<bool>(), count, val);
        break;
      }
      case dtype::float64: {
        double val = *value_.data_ptr<double>();
        std::fill_n(output.data_ptr<double>(), count, val);
        break;
      }
      default:
        throw std::runtime_error("constant_of_shape: unsupported dtype");
    }

    return output;
  }

 private:
  dynamic_tensor value_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::constant_of_shape_node, coalsack::graph_node)
