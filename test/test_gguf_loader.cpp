#include <chrono>
#include <iostream>

#include "gguf_loader.h"

using namespace coalsack;

int main(int argc, char** argv) {
  std::cout << "Testing GGUF Loader\n";
  std::cout << "===================\n\n";

  std::string model_path = "/workspaces/stargazer/models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "Loading GGUF: " << model_path << "\n";

  auto start = std::chrono::high_resolution_clock::now();

  gguf_loader loader;
  if (!loader.load(model_path)) {
    std::cerr << "Failed to load GGUF file\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "✓ Model loaded successfully in " << duration.count() << " ms\n\n";

  // Test: Basic metadata
  std::cout << "Basic Information:\n";
  std::cout << "  Version: " << loader.get_version() << "\n";
  std::cout << "  Tensor count: " << loader.get_tensor_count() << "\n";
  std::cout << "  Metadata entries: " << loader.get_kv_count() << "\n\n";

  // Test: Model architecture metadata
  std::cout << "Model Architecture:\n";

  auto arch = loader.get_string("general.architecture");
  if (arch) {
    std::cout << "  Architecture: " << *arch << "\n";
  }

  auto name = loader.get_string("general.name");
  if (name) {
    std::cout << "  Name: " << *name << "\n";
  }

  auto block_count = loader.get_uint32("gpt-oss.block_count");
  if (block_count) {
    std::cout << "  Block count: " << *block_count << "\n";
    if (*block_count != 24) {
      std::cerr << "  ERROR: Expected 24 blocks, got " << *block_count << "\n";
      return 1;
    }
  }

  auto embedding_length = loader.get_uint32("gpt-oss.embedding_length");
  if (embedding_length) {
    std::cout << "  Embedding length: " << *embedding_length << "\n";
    if (*embedding_length != 2880) {
      std::cerr << "  ERROR: Expected 2880 embedding length, got " << *embedding_length << "\n";
      return 1;
    }
  }

  auto context_length = loader.get_uint32("gpt-oss.context_length");
  if (context_length) {
    std::cout << "  Context length: " << *context_length << "\n";
  }

  std::cout << "\n";

  // Test: Attention parameters
  std::cout << "Attention Parameters:\n";

  auto head_count = loader.get_uint32("gpt-oss.attention.head_count");
  if (head_count) {
    std::cout << "  Q heads: " << *head_count << "\n";
  }

  auto head_count_kv = loader.get_uint32("gpt-oss.attention.head_count_kv");
  if (head_count_kv) {
    std::cout << "  KV heads: " << *head_count_kv << "\n";
  }

  std::cout << "\n";

  // Test: MoE parameters
  std::cout << "MoE Parameters:\n";

  auto expert_count = loader.get_uint32("gpt-oss.expert_count");
  if (expert_count) {
    std::cout << "  Expert count: " << *expert_count << "\n";
    if (*expert_count != 32) {
      std::cerr << "  ERROR: Expected 32 experts, got " << *expert_count << "\n";
      return 1;
    }
  }

  auto expert_used_count = loader.get_uint32("gpt-oss.expert_used_count");
  if (expert_used_count) {
    std::cout << "  Experts used (top-k): " << *expert_used_count << "\n";
    if (*expert_used_count != 4) {
      std::cerr << "  ERROR: Expected top-4 experts, got " << *expert_used_count << "\n";
      return 1;
    }
  }

  std::cout << "\n";

  // Test: Tokenizer
  std::cout << "Tokenizer:\n";

  auto tokenizer_model = loader.get_string("tokenizer.ggml.model");
  if (tokenizer_model) {
    std::cout << "  Model: " << *tokenizer_model << "\n";
  }

  auto tokens = loader.get_array_string("tokenizer.ggml.tokens");
  std::cout << "  Vocab size: " << tokens.size() << "\n";
  if (tokens.size() != 201088) {
    std::cerr << "  ERROR: Expected 201088 tokens, got " << tokens.size() << "\n";
    return 1;
  }

  auto merges = loader.get_array_string("tokenizer.ggml.merges");
  std::cout << "  BPE merges: " << merges.size() << "\n";

  std::cout << "\n";

  // Test: Tensor information
  std::cout << "Tensor Information:\n";

  auto tensor_names = loader.get_tensor_names();
  std::cout << "  Total tensors: " << tensor_names.size() << "\n";

  if (tensor_names.size() != 459) {
    std::cerr << "  ERROR: Expected 459 tensors, got " << tensor_names.size() << "\n";
    return 1;
  }

  // Check specific tensor
  auto token_embd_info = loader.get_tensor_info("token_embd.weight");
  if (token_embd_info) {
    std::cout << "  token_embd.weight: ";
    std::cout << "shape [";
    for (size_t i = 0; i < token_embd_info->shape.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << token_embd_info->shape[i];
    }
    std::cout << "], ";
    std::cout << "type " << ggml_type_name(token_embd_info->type);
    std::cout << ", size " << token_embd_info->size << " bytes\n";
  } else {
    std::cerr << "  ERROR: token_embd.weight not found\n";
    return 1;
  }

  // Show first few tensor names
  std::cout << "\n  First 10 tensors:\n";
  for (size_t i = 0; i < std::min<size_t>(10, tensor_names.size()); ++i) {
    auto info = loader.get_tensor_info(tensor_names[i]);
    if (info) {
      std::cout << "    " << (i + 1) << ". " << tensor_names[i];
      std::cout << " [";
      for (size_t j = 0; j < info->shape.size(); ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << info->shape[j];
      }
      std::cout << "] " << ggml_type_name(info->type) << "\n";
    }
  }

  std::cout << "\n✓ All tests passed!\n";

  return 0;
}
