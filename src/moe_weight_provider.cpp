#include "moe_weight_provider.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "gguf_dequant.h"
#include "gguf_multi_loader.h"

namespace coalsack {

namespace {

void dequantize_q4k_to_fp16(const uint8_t* src, uint16_t* dst, size_t num_elements) {
  std::vector<float> temp(num_elements);
  dequantize_block_q4_K(src, temp.data(), num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    dst[i] = fp32_to_fp16(temp[i]);
  }
}

}  // namespace

moe_weight_provider::moe_weight_provider(std::shared_ptr<gguf_multi_loader> loader,
                                         const std::vector<std::string>& shard_paths,
                                         size_t max_cache_size_bytes)
    : loader_(std::move(loader)),
      shard_paths_(shard_paths),
      current_cache_size_bytes_(0),
      max_cache_size_bytes_(max_cache_size_bytes),
      cache_hits_(0),
      cache_misses_(0) {
  if (!loader_) {
    throw std::invalid_argument("loader cannot be null");
  }

  if (shard_paths_.empty()) {
    throw std::invalid_argument("shard_paths cannot be empty");
  }

  shard_files_.reserve(shard_paths_.size());
  for (const auto& path : shard_paths_) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
      for (FILE* f : shard_files_) {
        if (f) std::fclose(f);
      }
      throw std::runtime_error("Failed to open shard file: " + path);
    }
    shard_files_.push_back(fp);
  }
}

moe_weight_provider::~moe_weight_provider() {
  for (FILE* fp : shard_files_) {
    if (fp) {
      std::fclose(fp);
    }
  }
}

std::variant<dynamic_tensor, dynamic_mx_tensor> moe_weight_provider::get(
    const std::string& tensor_name, int expert_id) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  std::string cache_key = tensor_name + ":" + std::to_string(expert_id);

  auto it = cache_.find(cache_key);
  if (it != cache_.end()) {
    ++cache_hits_;
    update_lru(cache_key);
    return it->second.tensor;
  }

  ++cache_misses_;

  auto tensor_info_opt = loader_->get_tensor_info(tensor_name);
  if (!tensor_info_opt) {
    throw std::runtime_error("Tensor not found: " + tensor_name);
  }

  auto& tensor_info = *tensor_info_opt;

  std::vector<size_t> reversed_shape;
  for (int i = tensor_info.shape.size() - 1; i >= 0; --i) {
    reversed_shape.push_back(tensor_info.shape[i]);
  }

  if (reversed_shape.size() < 2 || reversed_shape.size() > 3) {
    throw std::runtime_error("Expected 2D or 3D tensor for MoE weights, got " +
                             std::to_string(reversed_shape.size()) + "D");
  }

  size_t num_experts = reversed_shape[0];
  size_t dim0 = reversed_shape[1];
  size_t dim1 = (reversed_shape.size() == 3) ? reversed_shape[2] : 1;
  bool is_1d_output = (reversed_shape.size() == 2);

  if (static_cast<size_t>(expert_id) >= num_experts) {
    throw std::out_of_range("expert_id " + std::to_string(expert_id) + " out of range [0, " +
                            std::to_string(num_experts) + ")");
  }

  size_t slice_elements = dim0 * dim1;

  size_t slice_bytes_result = 0;
  if (tensor_info.type == ggml_type::Q4_K || tensor_info.type == ggml_type::MXFP4 ||
      tensor_info.type == ggml_type::F16) {
    slice_bytes_result = slice_elements * sizeof(uint16_t);
  } else if (tensor_info.type == ggml_type::F32) {
    slice_bytes_result = slice_elements * sizeof(float);
  } else {
    throw std::runtime_error("Unsupported tensor type: " +
                             std::to_string(static_cast<int>(tensor_info.type)));
  }

  evict_lru_until_space(slice_bytes_result);

  dynamic_tensor result;

  if (tensor_info.type == ggml_type::Q4_K) {
    constexpr size_t QK_K = 256;
    constexpr size_t Q4_K_BLOCK_SIZE = 144;

    size_t num_blocks = (slice_elements + QK_K - 1) / QK_K;
    size_t slice_bytes_quantized = num_blocks * Q4_K_BLOCK_SIZE;

    uint64_t expert_offset = tensor_info.offset + expert_id * slice_bytes_quantized;

    auto tensor_data =
        load_expert_slice_from_file(tensor_info.shard_idx, expert_offset, slice_bytes_quantized);

    std::vector<uint16_t> fp16_data(slice_elements);
    dequantize_q4k_to_fp16(tensor_data.data(), fp16_data.data(), slice_elements);

    if (is_1d_output) {
      result = dynamic_tensor(dtype::float16, {static_cast<int64_t>(dim0)});
    } else {
      result =
          dynamic_tensor(dtype::float16, {static_cast<int64_t>(dim0), static_cast<int64_t>(dim1)});
    }
    std::memcpy(result.data_ptr<uint16_t>(), fp16_data.data(), slice_bytes_result);
  } else if (tensor_info.type == ggml_type::MXFP4) {
    constexpr size_t MXFP4_BLOCK_SIZE = 17;

    size_t num_blocks = (slice_elements + 31) / 32;
    size_t slice_bytes_quantized = num_blocks * MXFP4_BLOCK_SIZE;

    uint64_t expert_offset = tensor_info.offset + expert_id * slice_bytes_quantized;

    dynamic_mx_tensor mx_result;
    if (is_1d_output) {
      mx_result = dynamic_mx_tensor({static_cast<int64_t>(dim0)});
    } else {
      mx_result = dynamic_mx_tensor({static_cast<int64_t>(dim0), static_cast<int64_t>(dim1)});
    }

    // Load directly into tensor storage
    load_expert_slice_from_file(tensor_info.shard_idx, expert_offset, mx_result.data_ptr(),
                                slice_bytes_quantized);

    cache_entry entry;
    entry.tensor = mx_result;
    entry.size_bytes = slice_bytes_quantized;

    cache_[cache_key] = entry;
    cache_lru_list_.push_back(cache_key);
    cache_lru_map_[cache_key] = std::prev(cache_lru_list_.end());
    current_cache_size_bytes_ += slice_bytes_quantized;

    return mx_result;
  } else if (tensor_info.type == ggml_type::F32) {
    uint64_t expert_offset = tensor_info.offset + expert_id * slice_bytes_result;
    auto tensor_data =
        load_expert_slice_from_file(tensor_info.shard_idx, expert_offset, slice_bytes_result);

    if (is_1d_output) {
      result = dynamic_tensor(dtype::float32, {static_cast<int64_t>(dim0)});
    } else {
      result =
          dynamic_tensor(dtype::float32, {static_cast<int64_t>(dim0), static_cast<int64_t>(dim1)});
    }
    std::memcpy(result.data_ptr<float>(), tensor_data.data(), slice_bytes_result);
  } else if (tensor_info.type == ggml_type::F16) {
    uint64_t expert_offset = tensor_info.offset + expert_id * slice_bytes_result;
    auto tensor_data =
        load_expert_slice_from_file(tensor_info.shard_idx, expert_offset, slice_bytes_result);

    if (is_1d_output) {
      result = dynamic_tensor(dtype::float16, {static_cast<int64_t>(dim0)});
    } else {
      result =
          dynamic_tensor(dtype::float16, {static_cast<int64_t>(dim0), static_cast<int64_t>(dim1)});
    }
    std::memcpy(result.data_ptr<uint16_t>(), tensor_data.data(), slice_bytes_result);
  }

  cache_entry entry;
  entry.tensor = result;
  entry.size_bytes = slice_bytes_result;

  cache_[cache_key] = entry;
  cache_lru_list_.push_back(cache_key);
  cache_lru_map_[cache_key] = std::prev(cache_lru_list_.end());
  current_cache_size_bytes_ += slice_bytes_result;

  return result;
}

void moe_weight_provider::set_max_cache_size(size_t max_bytes) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  max_cache_size_bytes_ = max_bytes;

  if (current_cache_size_bytes_ > max_cache_size_bytes_) {
    evict_lru_until_space(0);
  }
}

size_t moe_weight_provider::get_max_cache_size() const { return max_cache_size_bytes_; }

size_t moe_weight_provider::get_current_cache_size() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return current_cache_size_bytes_;
}

double moe_weight_provider::cache_hit_rate() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  size_t total = cache_hits_ + cache_misses_;
  if (total == 0) return 0.0;
  return static_cast<double>(cache_hits_) / total;
}

void moe_weight_provider::evict_lru_until_space(size_t required_bytes) {
  while (current_cache_size_bytes_ + required_bytes > max_cache_size_bytes_ &&
         !cache_lru_list_.empty()) {
    std::string lru_key = cache_lru_list_.front();
    cache_lru_list_.pop_front();
    cache_lru_map_.erase(lru_key);

    auto it = cache_.find(lru_key);
    if (it != cache_.end()) {
      current_cache_size_bytes_ -= it->second.size_bytes;
      cache_.erase(it);
    }
  }
}

void moe_weight_provider::update_lru(const std::string& key) {
  auto map_it = cache_lru_map_.find(key);
  if (map_it != cache_lru_map_.end()) {
    cache_lru_list_.splice(cache_lru_list_.end(), cache_lru_list_, map_it->second);
    map_it->second = std::prev(cache_lru_list_.end());
  }
}

std::vector<uint8_t> moe_weight_provider::load_expert_slice_from_file(uint32_t shard_idx,
                                                                      uint64_t offset,
                                                                      size_t num_bytes) {
  if (shard_idx >= shard_files_.size()) {
    throw std::runtime_error("Shard index " + std::to_string(shard_idx) + " out of range [0, " +
                             std::to_string(shard_files_.size()) + ")");
  }

  FILE* fp = shard_files_[shard_idx];

  if (std::fseek(fp, offset, SEEK_SET) != 0) {
    throw std::runtime_error("Failed to seek to offset " + std::to_string(offset) + " in shard " +
                             std::to_string(shard_idx));
  }

  std::vector<uint8_t> buffer(num_bytes);
  size_t bytes_read = std::fread(buffer.data(), 1, num_bytes, fp);

  if (bytes_read != num_bytes) {
    throw std::runtime_error("Failed to read " + std::to_string(num_bytes) + " bytes, got " +
                             std::to_string(bytes_read));
  }

  return buffer;
}

void moe_weight_provider::load_expert_slice_from_file(uint32_t shard_idx, uint64_t offset,
                                                      uint8_t* buffer, size_t num_bytes) {
  if (shard_idx >= shard_files_.size()) {
    throw std::runtime_error("Shard index " + std::to_string(shard_idx) + " out of range [0, " +
                             std::to_string(shard_files_.size()) + ")");
  }

  FILE* fp = shard_files_[shard_idx];

  if (std::fseek(fp, offset, SEEK_SET) != 0) {
    throw std::runtime_error("Failed to seek to offset " + std::to_string(offset) + " in shard " +
                             std::to_string(shard_idx));
  }

  size_t bytes_read = std::fread(buffer, 1, num_bytes, fp);

  if (bytes_read != num_bytes) {
    throw std::runtime_error("Failed to read " + std::to_string(num_bytes) + " bytes, got " +
                             std::to_string(bytes_read));
  }
}

}  // namespace coalsack
