#pragma once

#include <cstdio>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "dynamic_mx_tensor.h"
#include "dynamic_tensor.h"
#include "gguf_multi_loader.h"

namespace coalsack {

class moe_weight_provider {
 private:
  struct cache_entry {
    std::variant<dynamic_tensor, dynamic_mx_tensor> tensor;
    size_t size_bytes;
  };

  std::shared_ptr<gguf_multi_loader> loader_;
  std::vector<FILE*> shard_files_;
  std::vector<std::string> shard_paths_;
  
  std::unordered_map<std::string, cache_entry> cache_;
  std::list<std::string> cache_lru_list_;
  std::unordered_map<std::string, std::list<std::string>::iterator> cache_lru_map_;
  
  size_t current_cache_size_bytes_;
  size_t max_cache_size_bytes_;
  
  size_t cache_hits_;
  size_t cache_misses_;
  
  mutable std::mutex cache_mutex_;

 public:
  explicit moe_weight_provider(
      std::shared_ptr<gguf_multi_loader> loader,
      const std::vector<std::string>& shard_paths,
      size_t max_cache_size_bytes = 1024ULL * 1024 * 1024);
  
  ~moe_weight_provider();
  
  moe_weight_provider(const moe_weight_provider&) = delete;
  moe_weight_provider& operator=(const moe_weight_provider&) = delete;
  
  std::variant<dynamic_tensor, dynamic_mx_tensor> get(const std::string& tensor_name, int expert_id);
  
  void set_max_cache_size(size_t max_bytes);
  size_t get_max_cache_size() const;
  size_t get_current_cache_size() const;
  double cache_hit_rate() const;

 private:
  void evict_lru_until_space(size_t required_bytes);
  void update_lru(const std::string& key);
  std::vector<uint8_t> load_expert_slice_from_file(uint32_t shard_idx, uint64_t offset, size_t num_bytes);
  void load_expert_slice_from_file(uint32_t shard_idx, uint64_t offset, uint8_t* buffer, size_t num_bytes);
};

}  // namespace coalsack
