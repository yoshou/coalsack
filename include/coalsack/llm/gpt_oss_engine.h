#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/tensor/dynamic_tensor.h"

namespace coalsack {

// Forward declarations
class gguf_loader;
class gpt2_tokenizer;
class subgraph;
class graph_proc;

class gpt_oss_engine {
 public:
  struct config {
    int64_t kv_cache_size =
        std::numeric_limits<int64_t>::max();   // default: unlimited (use model's max_seq_len)
    size_t moe_cache_size_bytes = 2147483648;  // 2 GiB default
  };

  gpt_oss_engine();
  explicit gpt_oss_engine(const config& cfg);
  ~gpt_oss_engine();

  // Load model from GGUF file(s)
  bool load(const std::string& gguf_path);
  bool load(const std::vector<std::string>& gguf_paths);

  // Generate text from prompt
  // Returns generated text (without prompt)
  std::string generate(const std::string& prompt, size_t max_tokens = 128,
                       float temperature = 0.7f);

  // Check if model is loaded
  bool is_loaded() const;

  // Get model info
  int64_t get_vocab_size() const;
  int64_t get_num_layers() const;
  int64_t get_hidden_dim() const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl_;

  // Load model config from GGUF metadata
  void load_config_from_gguf();

  // Load weights from GGUF
  void load_weights_from_gguf();

  // Build the inference graph
  void build_transformer_graph();

  // Wire I/O nodes to the graph
  void wire_io_nodes(std::shared_ptr<class graph_edge> input_placeholder,
                     std::shared_ptr<class graph_edge> logits_output);

  // Sample next token from logits
  uint32_t sample_token(const float* logits, int64_t vocab_size, float temperature);

  // KV cache management
  void initialize_kv_caches();
  void reset_kv_caches();
};

}  // namespace coalsack
