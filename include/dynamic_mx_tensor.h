#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cereal/types/vector.hpp>

namespace coalsack {

// Storage class for MXFP4 quantized tensors
// MXFP4 format: 32 elements per block (17 bytes: 1 E8M0 scale + 16 packed 4-bit values)
class dynamic_mx_tensor_storage {
 public:
  static constexpr size_t ELEMENTS_PER_BLOCK = 32;
  static constexpr size_t BYTES_PER_BLOCK = 17;  // 1 E8M0 scale + 16 packed nibbles

  dynamic_mx_tensor_storage() = default;

  explicit dynamic_mx_tensor_storage(size_t num_blocks)
      : num_blocks_(num_blocks), data_(num_blocks * BYTES_PER_BLOCK) {}

  size_t num_blocks() const { return num_blocks_; }
  size_t size_bytes() const { return data_.size(); }
  size_t num_elements() const { return num_blocks_ * ELEMENTS_PER_BLOCK; }

  uint8_t* data() { return data_.data(); }
  const uint8_t* data() const { return data_.data(); }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(num_blocks_, data_);
  }

 private:
  size_t num_blocks_ = 0;
  std::vector<uint8_t> data_;
};

// Dynamic MXFP4 tensor class
class dynamic_mx_tensor {
 public:
  dynamic_mx_tensor() = default;

  explicit dynamic_mx_tensor(const std::vector<int64_t>& shape) : shape_(shape) {
    size_t total_elements = 1;
    for (auto dim : shape) {
      if (dim < 0) {
        throw std::invalid_argument("dynamic_mx_tensor: negative dimension");
      }
      total_elements *= dim;
    }
    num_elements_ = total_elements;
    
    size_t num_blocks = calc_num_blocks(total_elements);
    storage_ = std::make_shared<dynamic_mx_tensor_storage>(num_blocks);
  }

  static size_t calc_num_blocks(size_t num_elements) {
    return (num_elements + dynamic_mx_tensor_storage::ELEMENTS_PER_BLOCK - 1) /
           dynamic_mx_tensor_storage::ELEMENTS_PER_BLOCK;
  }

  size_t ndim() const { return shape_.size(); }
  
  int64_t dim(size_t idx) const {
    if (idx >= shape_.size()) {
      throw std::out_of_range("dynamic_mx_tensor: dimension index out of range");
    }
    return shape_[idx];
  }

  const std::vector<int64_t>& shape() const { return shape_; }
  
  size_t num_elements() const { return num_elements_; }
  
  size_t size_bytes() const {
    return storage_ ? storage_->size_bytes() : 0;
  }

  uint8_t* data_ptr() {
    if (!storage_) {
      throw std::runtime_error("dynamic_mx_tensor: null storage");
    }
    return storage_->data();
  }

  const uint8_t* data_ptr() const {
    if (!storage_) {
      throw std::runtime_error("dynamic_mx_tensor: null storage");
    }
    return storage_->data();
  }

  size_t num_blocks() const {
    return storage_ ? storage_->num_blocks() : 0;
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(shape_, num_elements_, storage_);
  }

 private:
  std::vector<int64_t> shape_;
  size_t num_elements_ = 0;
  std::shared_ptr<dynamic_mx_tensor_storage> storage_;
};

}  // namespace coalsack
