#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace coalsack {

// Dynamic tensor data type enumeration
enum class dtype {
  float32,
  int64,
  int32,
  bool_,
  uint8,
  float64,
};

inline size_t dtype_size(dtype dt) {
  switch (dt) {
    case dtype::float32:
      return sizeof(float);
    case dtype::int64:
      return sizeof(int64_t);
    case dtype::int32:
      return sizeof(int32_t);
    case dtype::bool_:
      return sizeof(bool);
    case dtype::uint8:
      return sizeof(uint8_t);
    case dtype::float64:
      return sizeof(double);
    default:
      throw std::invalid_argument("Unknown dtype");
  }
}

inline std::string dtype_name(dtype dt) {
  switch (dt) {
    case dtype::float32:
      return "float32";
    case dtype::int64:
      return "int64";
    case dtype::int32:
      return "int32";
    case dtype::bool_:
      return "bool";
    case dtype::uint8:
      return "uint8";
    case dtype::float64:
      return "float64";
    default:
      return "unknown";
  }
}

// Storage class with reference counting
class dynamic_tensor_storage {
 private:
  std::shared_ptr<std::vector<uint8_t>> data_;

 public:
  dynamic_tensor_storage() : data_(std::make_shared<std::vector<uint8_t>>()) {}

  explicit dynamic_tensor_storage(size_t size_bytes)
      : data_(std::make_shared<std::vector<uint8_t>>(size_bytes)) {}

  dynamic_tensor_storage(const void* data, size_t size_bytes)
      : data_(std::make_shared<std::vector<uint8_t>>(size_bytes)) {
    std::memcpy(data_->data(), data, size_bytes);
  }

  uint8_t* data() { return data_->data(); }
  const uint8_t* data() const { return data_->data(); }
  size_t size() const { return data_->size(); }

  void resize(size_t new_size) { data_->resize(new_size); }

  dynamic_tensor_storage clone() const {
    dynamic_tensor_storage result;
    result.data_ = std::make_shared<std::vector<uint8_t>>(*data_);
    return result;
  }

  long use_count() const { return data_.use_count(); }
};

// Dynamic tensor class with rank-variable shape
class dynamic_tensor {
 private:
  dtype dtype_;
  std::vector<int64_t> shape_;
  dynamic_tensor_storage storage_;

 public:
  dynamic_tensor() : dynamic_tensor(dtype::float32, {}) {}

  dynamic_tensor(dtype dt, const std::vector<int64_t>& shape)
      : dtype_(dt), shape_(shape), storage_(numel(shape) * dtype_size(dt)) {}

  dynamic_tensor(dtype dt, const std::vector<int64_t>& shape, const void* data)
      : dtype_(dt), shape_(shape), storage_(data, numel(shape) * dtype_size(dt)) {}

  dtype get_dtype() const { return dtype_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }

  int64_t dim(int64_t i) const {
    if (i < 0) i += ndim();
    if (i < 0 || i >= ndim()) {
      throw std::out_of_range("Dimension index out of range");
    }
    return shape_[i];
  }

  int64_t numel() const { return numel(shape_); }

  static int64_t numel(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
  }

  size_t bytes() const { return numel() * dtype_size(dtype_); }

  uint8_t* data() { return storage_.data(); }
  const uint8_t* data() const { return storage_.data(); }

  template <typename T>
  T* data_ptr() {
    return reinterpret_cast<T*>(storage_.data());
  }

  template <typename T>
  const T* data_ptr() const {
    return reinterpret_cast<const T*>(storage_.data());
  }

  dynamic_tensor reshape(const std::vector<int64_t>& new_shape) const {
    int64_t new_numel = numel(new_shape);
    if (new_numel != numel()) {
      throw std::invalid_argument(
          "Reshape: total size mismatch (current=" + std::to_string(numel()) +
          ", new=" + std::to_string(new_numel) + ")");
    }
    dynamic_tensor result;
    result.dtype_ = dtype_;
    result.shape_ = new_shape;
    result.storage_ = storage_;
    return result;
  }

  dynamic_tensor clone() const {
    dynamic_tensor result;
    result.dtype_ = dtype_;
    result.shape_ = shape_;
    result.storage_ = storage_.clone();
    return result;
  }

  // Broadcast shape check (returns true if this tensor can be broadcast to target_shape)
  bool can_broadcast_to(const std::vector<int64_t>& target_shape) const {
    if (ndim() > static_cast<int64_t>(target_shape.size())) return false;

    int64_t offset = target_shape.size() - ndim();
    for (int64_t i = 0; i < ndim(); ++i) {
      int64_t dim_a = shape_[i];
      int64_t dim_b = target_shape[offset + i];
      if (dim_a != dim_b && dim_a != 1) {
        return false;
      }
    }
    return true;
  }

  static std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& shape_a,
                                              const std::vector<int64_t>& shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
      int64_t dim_a = (i < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
      int64_t dim_b = (i < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;

      if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
        if (dim_a == 0 || dim_b == 0) {
          result[max_ndim - 1 - i] = 0;
        } else {
          result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        }
      } else {
        throw std::invalid_argument("Incompatible shapes for broadcasting");
      }
    }
    return result;
  }

  std::vector<int64_t> compute_strides() const {
    std::vector<int64_t> strides(shape_.size());
    if (shape_.empty()) return strides;

    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape_[i + 1];
    }
    return strides;
  }

  bool empty() const { return numel() == 0; }

  std::string to_string() const {
    std::string result = "dynamic_tensor(dtype=" + dtype_name(dtype_) + ", shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (i > 0) result += ", ";
      result += std::to_string(shape_[i]);
    }
    result += "], numel=" + std::to_string(numel()) + ")";
    return result;
  }
};

}  // namespace coalsack
