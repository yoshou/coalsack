#pragma once

#include <cmath>

#include "../nn_op_node.h"

namespace coalsack {

class erf_node : public unary_op_node {
 private:
  template <typename T>
  dynamic_tensor erf_impl(const dynamic_tensor& input) {
    dynamic_tensor output(input.get_dtype(), input.shape());
    const T* input_data = input.data_ptr<T>();
    T* output_data = output.data_ptr<T>();
    int64_t n = input.numel();
    for (int64_t i = 0; i < n; ++i) {
      output_data[i] = std::erf(input_data[i]);
    }
    return output;
  }

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() == dtype::float32) {
      return erf_impl<float>(input);
    } else if (input.get_dtype() == dtype::float64) {
      return erf_impl<double>(input);
    } else {
      throw std::runtime_error("erf: unsupported dtype (only float32 and float64 supported)");
    }
  }

 public:
  erf_node() : unary_op_node("erf") {}
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::erf_node, coalsack::graph_node)
