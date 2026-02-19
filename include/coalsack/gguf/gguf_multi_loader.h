#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "coalsack/gguf/gguf_loader.h"

namespace coalsack {

// Multi-file GGUF loader that merges multiple shard files
class gguf_multi_loader {
 public:
  gguf_multi_loader();
  ~gguf_multi_loader();

  // Non-copyable
  gguf_multi_loader(const gguf_multi_loader&) = delete;
  gguf_multi_loader& operator=(const gguf_multi_loader&) = delete;

  // Load multiple GGUF files
  bool load(const std::vector<std::string>& paths);

  bool is_loaded() const;

  uint32_t get_version() const;
  uint64_t get_tensor_count() const;
  uint64_t get_kv_count() const;

  // Metadata accessors (type-specific)
  std::optional<std::string> get_string(const std::string& key) const;
  std::optional<uint32_t> get_uint32(const std::string& key) const;
  std::optional<uint64_t> get_uint64(const std::string& key) const;
  std::optional<int32_t> get_int32(const std::string& key) const;
  std::optional<int64_t> get_int64(const std::string& key) const;
  std::optional<float> get_float32(const std::string& key) const;
  std::optional<double> get_float64(const std::string& key) const;
  std::optional<bool> get_bool(const std::string& key) const;

  // Array accessors
  std::vector<std::string> get_array_string(const std::string& key) const;
  std::vector<uint32_t> get_array_uint32(const std::string& key) const;
  std::vector<int32_t> get_array_int32(const std::string& key) const;
  std::vector<float> get_array_float32(const std::string& key) const;

  std::vector<std::string> get_metadata_keys() const;
  std::vector<std::string> get_tensor_names() const;

  struct tensor_info {
    std::string name;
    std::vector<uint64_t> shape;
    ggml_type type;
    uint64_t offset;     // Byte offset in shard file
    size_t size;         // Size in bytes
    uint32_t shard_idx;  // Which shard file (0-indexed)
  };

  std::optional<tensor_info> get_tensor_info(const std::string& name) const;

  // Get list of shard file paths
  const std::vector<std::string>& get_shard_paths() const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};

}  // namespace coalsack
