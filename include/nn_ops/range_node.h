#pragma once

#include <cmath>
#include <vector>

#include "../nn_op_node.h"

namespace coalsack {

class range_node : public variadic_op_node {
 public:
  range_node() : variadic_op_node("range", 3) {}

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    dtype input_dtype = inputs[0].get_dtype();
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i].get_dtype() != input_dtype) {
        throw std::runtime_error("range: all inputs must have the same dtype");
      }
    }

    if (input_dtype != dtype::float32 && input_dtype != dtype::float64 &&
        input_dtype != dtype::int64 && input_dtype != dtype::int32) {
      throw std::runtime_error("range: unsupported dtype");
    }

    // Compute count using common logic
    double start_d = get_scalar<double>(inputs[0]);
    double limit_d = get_scalar<double>(inputs[1]);
    double delta_d = get_scalar<double>(inputs[2]);

    if (delta_d == 0.0) {
      throw std::runtime_error("range: delta cannot be 0");
    }

    int64_t count = static_cast<int64_t>(std::ceil((limit_d - start_d) / delta_d));
    if (count < 0) {
      count = 0;
    }

    // Generate output based on input dtype
    if (input_dtype == dtype::float32 || input_dtype == dtype::float64) {
      dynamic_tensor output(dtype::float32, {count});
      float* out_data = output.data_ptr<float>();
      float start = get_scalar<float>(inputs[0]);
      float delta = get_scalar<float>(inputs[2]);
      for (int64_t i = 0; i < count; ++i) {
        out_data[i] = start + i * delta;
      }
      return output;
    } else {
      dynamic_tensor output(dtype::int64, {count});
      int64_t* out_data = output.data_ptr<int64_t>();
      int64_t start = get_scalar<int64_t>(inputs[0]);
      int64_t delta = get_scalar<int64_t>(inputs[2]);
      for (int64_t i = 0; i < count; ++i) {
        out_data[i] = start + i * delta;
      }
      return output;
    }
  }

 private:
  template <typename T>
  T get_scalar(const dynamic_tensor& t) {
    if (t.get_dtype() == dtype::int64) {
      return static_cast<T>(*t.data_ptr<int64_t>());
    }
    if (t.get_dtype() == dtype::int32) {
      return static_cast<T>(*t.data_ptr<int32_t>());
    }
    if (t.get_dtype() == dtype::float32) {
      return static_cast<T>(*t.data_ptr<float>());
    }
    if (t.get_dtype() == dtype::float64) {
      return static_cast<T>(*t.data_ptr<double>());
    }
    throw std::runtime_error("range: unsupported input dtype");
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::range_node, coalsack::graph_node)
