#pragma once

#include <cstring>

#include "../nn_op_node.h"

namespace coalsack {

class cast_node : public unary_op_node {
 private:
  dtype target_dtype_;

 protected:
  dynamic_tensor compute(const dynamic_tensor& input) override {
    if (input.get_dtype() == target_dtype_) {
      return input.clone();
    }

    dynamic_tensor output(target_dtype_, input.shape());

    int64_t numel = input.numel();

    if (input.get_dtype() == dtype::float32) {
      const float* src = input.data_ptr<float>();

      if (target_dtype_ == dtype::int64) {
        int64_t* dst = output.data_ptr<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<int64_t>(src[i]);
        }
      } else if (target_dtype_ == dtype::int32) {
        int32_t* dst = output.data_ptr<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<int32_t>(src[i]);
        }
      } else if (target_dtype_ == dtype::bool_) {
        bool* dst = output.data_ptr<bool>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = (src[i] != 0.0f);
        }
      } else if (target_dtype_ == dtype::float64) {
        double* dst = output.data_ptr<double>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<double>(src[i]);
        }
      } else {
        throw std::runtime_error("cast: unsupported target dtype");
      }
    } else if (input.get_dtype() == dtype::float64) {
      const double* src = input.data_ptr<double>();

      if (target_dtype_ == dtype::float32) {
        float* dst = output.data_ptr<float>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<float>(src[i]);
        }
      } else if (target_dtype_ == dtype::int64) {
        int64_t* dst = output.data_ptr<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<int64_t>(src[i]);
        }
      } else if (target_dtype_ == dtype::int32) {
        int32_t* dst = output.data_ptr<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<int32_t>(src[i]);
        }
      } else if (target_dtype_ == dtype::bool_) {
        bool* dst = output.data_ptr<bool>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = (src[i] != 0.0);
        }
      } else {
        throw std::runtime_error("cast: unsupported target dtype");
      }
    } else if (input.get_dtype() == dtype::int64) {
      const int64_t* src = input.data_ptr<int64_t>();

      if (target_dtype_ == dtype::float32) {
        float* dst = output.data_ptr<float>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<float>(src[i]);
        }
      } else if (target_dtype_ == dtype::int32) {
        int32_t* dst = output.data_ptr<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<int32_t>(src[i]);
        }
      } else if (target_dtype_ == dtype::bool_) {
        bool* dst = output.data_ptr<bool>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = (src[i] != 0);
        }
      } else if (target_dtype_ == dtype::float64) {
        double* dst = output.data_ptr<double>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<double>(src[i]);
        }
      } else {
        throw std::runtime_error("cast: unsupported target dtype");
      }
    } else if (input.get_dtype() == dtype::bool_) {
      const bool* src = input.data_ptr<bool>();

      if (target_dtype_ == dtype::float32) {
        float* dst = output.data_ptr<float>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = src[i] ? 1.0f : 0.0f;
        }
      } else if (target_dtype_ == dtype::float64) {
        double* dst = output.data_ptr<double>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = src[i] ? 1.0 : 0.0;
        }
      } else if (target_dtype_ == dtype::int64) {
        int64_t* dst = output.data_ptr<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = src[i] ? 1 : 0;
        }
      } else if (target_dtype_ == dtype::int32) {
        int32_t* dst = output.data_ptr<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = src[i] ? 1 : 0;
        }
      } else {
        throw std::runtime_error("cast: unsupported target dtype");
      }
    } else if (input.get_dtype() == dtype::int32) {
      const int32_t* src = input.data_ptr<int32_t>();

      if (target_dtype_ == dtype::float32) {
        float* dst = output.data_ptr<float>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<float>(src[i]);
        }
      } else if (target_dtype_ == dtype::int64) {
        int64_t* dst = output.data_ptr<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<int64_t>(src[i]);
        }
      } else if (target_dtype_ == dtype::bool_) {
        bool* dst = output.data_ptr<bool>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = (src[i] != 0);
        }
      } else if (target_dtype_ == dtype::float64) {
        double* dst = output.data_ptr<double>();
        for (int64_t i = 0; i < numel; ++i) {
          dst[i] = static_cast<double>(src[i]);
        }
      } else {
        throw std::runtime_error("cast: unsupported target dtype");
      }
    } else {
      throw std::runtime_error("cast: unsupported source dtype");
    }

    return output;
  }

 public:
  cast_node() : unary_op_node("cast"), target_dtype_(dtype::float32) {}

  void set_target_dtype(dtype dt) { target_dtype_ = dt; }

  void set_target_dtype(int onnx_dtype) {
    switch (onnx_dtype) {
      case 1:
        target_dtype_ = dtype::float32;
        break;
      case 7:
        target_dtype_ = dtype::int64;
        break;
      case 6:
        target_dtype_ = dtype::int32;
        break;
      case 9:
        target_dtype_ = dtype::bool_;
        break;
      case 11:
        target_dtype_ = dtype::float64;
        break;
      default:
        throw std::runtime_error("cast: unsupported dtype: " + std::to_string(onnx_dtype));
    }
  }

  void set_to_dtype(int onnx_dtype) { set_target_dtype(onnx_dtype); }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::cast_node, coalsack::graph_node)
