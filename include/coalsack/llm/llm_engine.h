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
    std::vector<int> hidden_layer_indices;     // FFN-residual layers to capture
  };

  llm_engine();
  explicit llm_engine(const config& cfg);
  ~llm_engine();

  // Load model from an already-opened gguf_multi_loader
  void load(std::shared_ptr<gguf_multi_loader> loader);

  // Prefill with prompt tokens and start the graph processor
  void start(const std::vector<uint32_t>& prompt_tokens);

  // Decode one token
  void next(uint32_t token);

  // Decode multiple tokens in one batch (for speculative decoding verification)
  void next_batch(const std::vector<uint32_t>& tokens);

  // Rollback KV cache and position to a given sequence position
  void rollback_to(int64_t position);

  // Stop the graph processor
  void stop();

  // Check if model is loaded
  bool is_loaded() const;

  // Get model info
  int64_t get_num_layers() const;
  int64_t get_hidden_dim() const;

  // State accessors — valid after start() / next().
  const std::vector<float>& get_logits() const;
  // All positions: [seq_len * vocab_size] — valid after next_batch()
  const std::vector<float>& get_logits_all_pos() const;
  // Throws if layer_index was not in hidden_layer_indices.
  const std::vector<float>& get_hidden_layer(
      int layer_index) const;  // last token only [hidden_dim]
  const std::vector<float>& get_hidden_layer_all_pos(
      int layer_index) const;  // all positions [seq_len * hidden_dim]

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
  void run_inference_step(const std::vector<uint32_t>& tokens);

  // KV cache management
  void initialize_kv_caches();
  void reset_kv_caches();
};

}  // namespace coalsack
