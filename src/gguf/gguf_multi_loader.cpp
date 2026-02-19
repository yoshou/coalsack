#include "coalsack/gguf/gguf_multi_loader.h"

#include <iostream>
#include <unordered_map>

namespace coalsack {

struct gguf_multi_loader::impl {
  std::vector<std::unique_ptr<gguf_loader>> loaders;
  std::vector<std::string> shard_paths;
  bool loaded = false;

  uint32_t version = 0;
  uint64_t total_tensor_count = 0;
  uint64_t total_kv_count = 0;

  // Mapping: tensor name -> shard index
  std::unordered_map<std::string, size_t> tensor_to_shard;
};

gguf_multi_loader::gguf_multi_loader() : pimpl_(std::make_unique<impl>()) {}

gguf_multi_loader::~gguf_multi_loader() = default;

bool gguf_multi_loader::load(const std::vector<std::string>& paths) {
  if (paths.empty()) {
    std::cerr << "Error: No GGUF files specified\n";
    return false;
  }

  pimpl_->shard_paths = paths;
  pimpl_->loaders.clear();
  pimpl_->tensor_to_shard.clear();
  pimpl_->total_tensor_count = 0;
  pimpl_->total_kv_count = 0;
  pimpl_->loaded = false;

  // Load each shard file
  for (size_t i = 0; i < paths.size(); ++i) {
    auto loader = std::make_unique<gguf_loader>();

    if (!loader->load(paths[i])) {
      std::cerr << "Error: Failed to load shard " << i << ": " << paths[i] << "\n";
      return false;
    }

    // Get version from first shard
    if (i == 0) {
      pimpl_->version = loader->get_version();
    } else {
      // Verify version matches
      if (loader->get_version() != pimpl_->version) {
        std::cerr << "Warning: Version mismatch in shard " << i << ": " << loader->get_version()
                  << " (expected " << pimpl_->version << ")\n";
      }
    }

    // Accumulate counts
    pimpl_->total_kv_count += loader->get_kv_count();
    pimpl_->total_tensor_count += loader->get_tensor_count();

    // Map tensors to shard index
    auto tensor_names = loader->get_tensor_names();
    for (const auto& name : tensor_names) {
      if (pimpl_->tensor_to_shard.find(name) != pimpl_->tensor_to_shard.end()) {
        std::cerr << "Error: Duplicate tensor name '" << name << "' in shard " << i << "\n";
        return false;
      }
      pimpl_->tensor_to_shard[name] = i;
    }

    pimpl_->loaders.push_back(std::move(loader));
  }

  pimpl_->loaded = true;
  return true;
}

bool gguf_multi_loader::is_loaded() const { return pimpl_->loaded; }

uint32_t gguf_multi_loader::get_version() const { return pimpl_->version; }

uint64_t gguf_multi_loader::get_tensor_count() const { return pimpl_->total_tensor_count; }

uint64_t gguf_multi_loader::get_kv_count() const { return pimpl_->total_kv_count; }

std::optional<std::string> gguf_multi_loader::get_string(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_string(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<uint32_t> gguf_multi_loader::get_uint32(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_uint32(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<uint64_t> gguf_multi_loader::get_uint64(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_uint64(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<int32_t> gguf_multi_loader::get_int32(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_int32(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<int64_t> gguf_multi_loader::get_int64(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_int64(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<float> gguf_multi_loader::get_float32(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_float32(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<double> gguf_multi_loader::get_float64(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_float64(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::optional<bool> gguf_multi_loader::get_bool(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_bool(key);
    if (result) return result;
  }
  return std::nullopt;
}

std::vector<std::string> gguf_multi_loader::get_array_string(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_array_string(key);
    if (!result.empty()) return result;
  }
  return {};
}

std::vector<uint32_t> gguf_multi_loader::get_array_uint32(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_array_uint32(key);
    if (!result.empty()) return result;
  }
  return {};
}

std::vector<int32_t> gguf_multi_loader::get_array_int32(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_array_int32(key);
    if (!result.empty()) return result;
  }
  return {};
}

std::vector<float> gguf_multi_loader::get_array_float32(const std::string& key) const {
  for (const auto& loader : pimpl_->loaders) {
    auto result = loader->get_array_float32(key);
    if (!result.empty()) return result;
  }
  return {};
}

std::vector<std::string> gguf_multi_loader::get_metadata_keys() const {
  // Merge keys from all shards (order undefined, but that's OK)
  std::unordered_map<std::string, bool> seen;
  std::vector<std::string> all_keys;

  for (const auto& loader : pimpl_->loaders) {
    auto keys = loader->get_metadata_keys();
    for (const auto& key : keys) {
      if (seen.find(key) == seen.end()) {
        seen[key] = true;
        all_keys.push_back(key);
      }
    }
  }

  return all_keys;
}

std::vector<std::string> gguf_multi_loader::get_tensor_names() const {
  // Collect all tensor names from all shards (order undefined, but that's OK)
  std::vector<std::string> all_names;
  all_names.reserve(pimpl_->total_tensor_count);

  for (const auto& loader : pimpl_->loaders) {
    auto names = loader->get_tensor_names();
    all_names.insert(all_names.end(), names.begin(), names.end());
  }

  return all_names;
}

std::optional<gguf_multi_loader::tensor_info> gguf_multi_loader::get_tensor_info(
    const std::string& name) const {
  auto it = pimpl_->tensor_to_shard.find(name);
  if (it == pimpl_->tensor_to_shard.end()) {
    return std::nullopt;
  }

  size_t shard_idx = it->second;
  auto info_opt = pimpl_->loaders[shard_idx]->get_tensor_info(name);
  if (!info_opt) {
    return std::nullopt;
  }

  // Convert to multi_loader tensor_info with shard_idx
  tensor_info result;
  result.name = info_opt->name;
  result.shape = info_opt->shape;
  result.type = info_opt->type;
  result.offset = info_opt->offset;
  result.size = info_opt->size;
  result.shard_idx = static_cast<uint32_t>(shard_idx);

  return result;
}

const std::vector<std::string>& gguf_multi_loader::get_shard_paths() const {
  return pimpl_->shard_paths;
}

}  // namespace coalsack
