#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/tensor/dynamic_tensor.h"

namespace coalsack {

// EAGLE3 speculative draft decoder (single transformer block + encoder FC).
class eagle3_speculative_decoder {
 public:
  struct config {
    int64_t max_seq_len = 2048;  // KV cache max sequence length
  };

  eagle3_speculative_decoder();
  explicit eagle3_speculative_decoder(const config& cfg);
  ~eagle3_speculative_decoder();

  // Load weights from GGUF eagle3 file
  bool load(const std::string& gguf_path);
  bool is_loaded() const;

  // Target model layers to capture (indices into llm_engine::config::hidden_layer_indices)
  const std::vector<int>& get_extract_layers() const;
  int64_t get_target_hidden_size() const;
  int64_t get_draft_vocab_size() const;

  // Compute g_embeddings from target model hidden states: concat(h_layers) → fc
  // all_hidden_states: layer_index → flat float32 [seq_len * target_hidden_size]
  // Returns [1, seq_len, eagle3_H]  (pre-normalization; decoder graph applies hidden_norm)
  dynamic_tensor encode(const std::unordered_map<int, std::vector<float>>& all_hidden_states,
                        int64_t seq_len) const;

  // Reset KV cache and start graph processor
  void start();

  // Run one decode step; g_embd is [1, n, eagle3_H], start_pos is KV cache offset
  void decode(const std::vector<uint32_t>& tokens, const dynamic_tensor& g_embd, int64_t start_pos);

  void stop();

  // State accessors — valid after decode()
  const dynamic_tensor& get_logits() const;  // [1, n, draft_vocab]
  const dynamic_tensor& get_prenorm()
      const;  // [1, n, eagle3_H]; last slice is g_embd for next step

  // Map draft vocab id to target vocab id via d2t delta table
  int64_t draft_to_target(int64_t draft_id) const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};

}  // namespace coalsack
