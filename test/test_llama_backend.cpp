#include <chrono>
#include <iostream>

#include "llama_backend.h"

using namespace coalsack;

int main(int argc, char** argv) {
  std::cout << "Testing llama.cpp Backend\n";
  std::cout << "=========================\n\n";

  std::string model_path = "/workspaces/stargazer/models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  auto start = std::chrono::high_resolution_clock::now();

  llama_backend backend;
  if (!backend.load(model_path, 2048, 0)) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "✓ Model loaded in " << duration.count() << " ms\n\n";

  std::cout << "Backend Info:\n";
  std::cout << "  Vocab size: " << backend.get_n_vocab() << "\n";
  std::cout << "  Context size: " << backend.get_n_ctx() << "\n";
  std::cout << "  Is loaded: " << (backend.is_loaded() ? "yes" : "no") << "\n\n";

  if (backend.get_n_vocab() != 201088) {
    std::cerr << "ERROR: Expected 201088 vocab size, got " << backend.get_n_vocab() << "\n";
    return 1;
  }

  if (backend.get_n_ctx() != 2048) {
    std::cerr << "ERROR: Expected 2048 context size, got " << backend.get_n_ctx() << "\n";
    return 1;
  }

  std::cout << "Test: Eval simple tokens\n";
  std::vector<uint32_t> test_tokens = {13225, 11, 2375};

  auto logits = backend.eval(test_tokens, 0);

  if (logits.empty()) {
    std::cerr << "ERROR: Eval returned empty logits\n";
    return 1;
  }

  std::cout << "  Logits size: " << logits.size() << "\n";

  if (logits.size() != static_cast<size_t>(backend.get_n_vocab())) {
    std::cerr << "ERROR: Logits size mismatch\n";
    return 1;
  }

  std::cout << "  ✓ Eval successful\n\n";

  std::cout << "Test: Reset\n";
  backend.reset();
  std::cout << "  ✓ Reset successful\n\n";

  std::cout << "✓ All tests passed!\n";

  return 0;
}
