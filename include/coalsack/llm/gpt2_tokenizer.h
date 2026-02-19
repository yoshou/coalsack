#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "coalsack/gguf/gguf_loader.h"
#include "coalsack/gguf/gguf_multi_loader.h"

namespace coalsack {

class gpt2_tokenizer {
 public:
  gpt2_tokenizer();
  ~gpt2_tokenizer();

  gpt2_tokenizer(const gpt2_tokenizer&) = delete;
  gpt2_tokenizer& operator=(const gpt2_tokenizer&) = delete;

  bool load_from_gguf(const gguf_loader& loader);
  bool load_from_gguf(const gguf_multi_loader& loader);

  std::vector<uint32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<uint32_t>& tokens) const;

  uint32_t bos_token_id() const;
  uint32_t eos_token_id() const;

  bool add_bos_token() const;
  bool add_eos_token() const;

  size_t vocab_size() const;

 private:
  template <typename LoaderType>
  bool load_from_gguf_impl(const LoaderType& loader);

  struct impl;
  std::unique_ptr<impl> pimpl_;
};

}  // namespace coalsack
