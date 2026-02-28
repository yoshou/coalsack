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
class gguf_multi_loader;
class subgraph;
class graph_proc;

class llm_engine {
 public:
  struct config {
    int64_t kv_cache_size =
        std::numeric_limits<int64_t>::max();   // default: unlimited (use model's max_seq_len)
    size_t moe_cache_size_bytes = 2147483648;  // 2 GiB default
  };

  llm_engine();
  explicit llm_engine(const config& cfg);
  ~llm_engine();

  // Load model from an already-opened gguf_multi_loader
  void load(std::shared_ptr<gguf_multi_loader> loader);

  // Prefill with prompt tokens and start the graph processor
  std::vector<float> start(const std::vector<uint32_t>& prompt_tokens);

  // Decode one token and return logits for the next token
  std::vector<float> next(uint32_t token);

  // Stop the graph processor
  void stop();

  // Check if model is loaded
  bool is_loaded() const;

  // Get model info
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

  // Single-step inference helper used by start() and next()
  std::vector<float> run_inference_step(const std::vector<uint32_t>& tokens);

  // KV cache management
  void initialize_kv_caches();
  void reset_kv_caches();
};

}  // namespace coalsack
