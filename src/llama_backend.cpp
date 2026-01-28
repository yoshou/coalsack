#include "llama_backend.h"

#include <llama.h>

#include <iostream>

namespace coalsack {

struct llama_backend::impl {
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  llama_memory_t memory = nullptr;
  bool loaded = false;
  int n_ctx = 0;
  int n_vocab = 0;

  impl() { ggml_backend_load_all(); }

  ~impl() {
    if (ctx) {
      llama_free(ctx);
    }
    if (model) {
      llama_model_free(model);
    }
  }
};

llama_backend::llama_backend() : pimpl_(std::make_unique<impl>()) {}

llama_backend::~llama_backend() = default;

bool llama_backend::load(const std::string& model_path, int n_ctx, int n_gpu_layers) {
  if (pimpl_->loaded) {
    std::cerr << "Error: Model already loaded\n";
    return false;
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = n_gpu_layers;

  pimpl_->model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!pimpl_->model) {
    std::cerr << "Error: Failed to load model from " << model_path << "\n";
    return false;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = n_ctx;
  ctx_params.n_batch = 512;
  ctx_params.n_threads = 4;
  ctx_params.no_perf = false;

  pimpl_->ctx = llama_init_from_model(pimpl_->model, ctx_params);
  if (!pimpl_->ctx) {
    std::cerr << "Error: Failed to create context\n";
    llama_model_free(pimpl_->model);
    pimpl_->model = nullptr;
    return false;
  }

  pimpl_->memory = llama_get_memory(pimpl_->ctx);
  pimpl_->n_ctx = n_ctx;
  const llama_vocab* vocab = llama_model_get_vocab(pimpl_->model);
  pimpl_->n_vocab = llama_vocab_n_tokens(vocab);
  pimpl_->loaded = true;

  std::cout << "Model loaded: " << pimpl_->n_vocab << " vocab, " << pimpl_->n_ctx
            << " context size\n";

  return true;
}

std::vector<uint32_t> llama_backend::tokenize(const std::string& text, bool add_special) const {
  if (!pimpl_->loaded) {
    std::cerr << "Error: Model not loaded\n";
    return {};
  }

  const llama_vocab* vocab = llama_model_get_vocab(pimpl_->model);
  int n_max_tokens = text.size() + (add_special ? 2 : 0);
  std::vector<llama_token> tokens(n_max_tokens);

  int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_max_tokens,
                                add_special,  // add_special
                                true          // parse_special - important for control tokens!
  );

  if (n_tokens < 0) {
    n_tokens = -n_tokens;
    tokens.resize(n_tokens);
    n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_tokens,
                              add_special, true);
  }

  tokens.resize(n_tokens);
  return std::vector<uint32_t>(tokens.begin(), tokens.end());
}

std::string llama_backend::detokenize(const std::vector<uint32_t>& tokens) const {
  if (!pimpl_->loaded) {
    std::cerr << "Error: Model not loaded\n";
    return "";
  }

  const llama_vocab* vocab = llama_model_get_vocab(pimpl_->model);
  std::vector<llama_token> llama_tokens(tokens.begin(), tokens.end());

  // Allocate buffer for detokenization
  int buffer_size = tokens.size() * 16 + 1;  // Conservative estimate
  std::vector<char> buffer(buffer_size);

  int result_len =
      llama_detokenize(vocab, llama_tokens.data(), llama_tokens.size(), buffer.data(), buffer_size,
                       false,  // remove_special
                       true    // unparse_special - important for control tokens!
      );

  if (result_len < 0) {
    // Buffer was too small, resize and retry
    buffer_size = -result_len;
    buffer.resize(buffer_size);
    result_len = llama_detokenize(vocab, llama_tokens.data(), llama_tokens.size(), buffer.data(),
                                  buffer_size, false, true);
  }

  return std::string(buffer.data(), result_len > 0 ? result_len : 0);
}

std::vector<float> llama_backend::eval(const std::vector<uint32_t>& tokens, int n_past) {
  if (!pimpl_->loaded) {
    std::cerr << "Error: Model not loaded\n";
    return {};
  }

  std::vector<llama_token> llama_tokens(tokens.begin(), tokens.end());
  const int n_tokens = llama_tokens.size();

  // Allocate batch (n_seq_max = 1 for single sequence)
  llama_batch batch = llama_batch_init(n_tokens, 0, 1);

  // Set tokens and positions
  batch.n_tokens = n_tokens;
  for (int i = 0; i < n_tokens; ++i) {
    batch.token[i] = llama_tokens[i];
    batch.pos[i] = n_past + i;
    batch.n_seq_id[i] = 1;   // Number of sequences this token belongs to
    batch.seq_id[i][0] = 0;  // Use sequence ID 0
    // Only request logits for the last token
    batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
  }

  int ret = llama_decode(pimpl_->ctx, batch);
  llama_batch_free(batch);

  if (ret != 0) {
    std::cerr << "Error: llama_decode failed with code " << ret << "\n";
    return {};
  }

  int n_vocab = pimpl_->n_vocab;
  // Use -1 to get logits for the last token
  float* logits = llama_get_logits_ith(pimpl_->ctx, -1);
  if (!logits) {
    std::cerr << "Error: failed to get logits\n";
    return {};
  }

  return std::vector<float>(logits, logits + n_vocab);
}

void llama_backend::reset() {
  if (pimpl_->memory) {
    llama_memory_seq_rm(pimpl_->memory, -1, 0, -1);
  }
}

bool llama_backend::is_loaded() const { return pimpl_->loaded; }

int llama_backend::get_n_vocab() const { return pimpl_->n_vocab; }

int llama_backend::get_n_ctx() const { return pimpl_->n_ctx; }

}  // namespace coalsack
