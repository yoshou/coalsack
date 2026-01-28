#pragma once

#include <memory>
#include <string>
#include <vector>

namespace coalsack {

class llama_backend {
 public:
  llama_backend();
  ~llama_backend();

  bool load(const std::string& model_path, int n_ctx = 2048, int n_gpu_layers = 0);

  std::vector<float> eval(const std::vector<uint32_t>& tokens, int n_past = 0);

  std::vector<uint32_t> tokenize(const std::string& text, bool add_special = false) const;
  std::string detokenize(const std::vector<uint32_t>& tokens) const;

  void reset();

  bool is_loaded() const;

  int get_n_vocab() const;
  int get_n_ctx() const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};

}  // namespace coalsack
