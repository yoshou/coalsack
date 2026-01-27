#pragma once

#include <cmath>
#include <limits>

#include "../nn_op_node.h"

namespace coalsack {

class clip_node : public variadic_op_node {
 private:
  template <typename T>
  dynamic_tensor clip_impl(const dynamic_tensor& input, T min_val, T max_val) {
    dynamic_tensor output(input.get_dtype(), input.shape());
    const T* in_data = input.data_ptr<T>();
    T* out_data = output.data_ptr<T>();
    int64_t n = input.numel();

    for (int64_t i = 0; i < n; ++i) {
      out_data[i] = std::min(std::max(in_data[i], min_val), max_val);
    }
    return output;
  }

 protected:
  dynamic_tensor compute(const std::vector<dynamic_tensor>& inputs) override {
    const auto& input = inputs[0];
    dtype dt = input.get_dtype();

    if (dt == dtype::float32) {
      float min_val =
          has_attr_min_ ? static_cast<float>(attr_min_) : -std::numeric_limits<float>::infinity();
      float max_val =
          has_attr_max_ ? static_cast<float>(attr_max_) : std::numeric_limits<float>::infinity();

      if (inputs.size() > 1 && !inputs[1].shape().empty() && inputs[1].numel() > 0) {
        min_val = *inputs[1].data_ptr<float>();
      }
      if (inputs.size() > 2 && !inputs[2].shape().empty() && inputs[2].numel() > 0) {
        max_val = *inputs[2].data_ptr<float>();
      }
      return clip_impl<float>(input, min_val, max_val);
    } else if (dt == dtype::float64) {
      double min_val = has_attr_min_ ? attr_min_ : -std::numeric_limits<double>::infinity();
      double max_val = has_attr_max_ ? attr_max_ : std::numeric_limits<double>::infinity();

      if (inputs.size() > 1 && !inputs[1].shape().empty() && inputs[1].numel() > 0) {
        min_val = *inputs[1].data_ptr<double>();
      }
      if (inputs.size() > 2 && !inputs[2].shape().empty() && inputs[2].numel() > 0) {
        max_val = *inputs[2].data_ptr<double>();
      }
      return clip_impl<double>(input, min_val, max_val);
    } else if (dt == dtype::int32) {
      int32_t min_val =
          has_attr_min_ ? static_cast<int32_t>(attr_min_) : std::numeric_limits<int32_t>::min();
      int32_t max_val =
          has_attr_max_ ? static_cast<int32_t>(attr_max_) : std::numeric_limits<int32_t>::max();

      if (inputs.size() > 1 && !inputs[1].shape().empty() && inputs[1].numel() > 0) {
        min_val = *inputs[1].data_ptr<int32_t>();
      }
      if (inputs.size() > 2 && !inputs[2].shape().empty() && inputs[2].numel() > 0) {
        max_val = *inputs[2].data_ptr<int32_t>();
      }
      return clip_impl<int32_t>(input, min_val, max_val);
    } else if (dt == dtype::int64) {
      int64_t min_val =
          has_attr_min_ ? static_cast<int64_t>(attr_min_) : std::numeric_limits<int64_t>::min();
      int64_t max_val =
          has_attr_max_ ? static_cast<int64_t>(attr_max_) : std::numeric_limits<int64_t>::max();

      if (inputs.size() > 1 && !inputs[1].shape().empty() && inputs[1].numel() > 0) {
        min_val = *inputs[1].data_ptr<int64_t>();
      }
      if (inputs.size() > 2 && !inputs[2].shape().empty() && inputs[2].numel() > 0) {
        max_val = *inputs[2].data_ptr<int64_t>();
      }
      return clip_impl<int64_t>(input, min_val, max_val);
    } else {
      throw std::runtime_error("clip: unsupported dtype");
    }
  }

 public:
  clip_node()
      : variadic_op_node("clip", 3),
        has_attr_min_(false),
        has_attr_max_(false),
        attr_min_(0),
        attr_max_(0) {}

  void set_min(double min) {
    attr_min_ = min;
    has_attr_min_ = true;
  }
  void set_max(double max) {
    attr_max_ = max;
    has_attr_max_ = true;
  }

 private:
  bool has_attr_min_, has_attr_max_;
  double attr_min_, attr_max_;
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::clip_node, coalsack::graph_node)
