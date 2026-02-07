#include <chrono>
#include <iostream>
#include <vector>

#include "gguf_loader.h"
#include "gguf_multi_loader.h"

using namespace coalsack;

int main(int argc, char** argv) {
  std::cout << "Testing GGUF Loader\n";
  std::cout << "===================\n\n";

  // Collect all file paths from command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <gguf_file1> [gguf_file2] ...\n";
    return 1;
  }

  std::vector<std::string> file_paths;
  for (int i = 1; i < argc; i++) {
    file_paths.push_back(argv[i]);
  }

  std::cout << "GGUF file(s): " << file_paths.size() << "\n";
  for (size_t i = 0; i < file_paths.size(); i++) {
    std::cout << "  [" << (i+1) << "] " << file_paths[i] << "\n";
  }
  std::cout << "\n";

  auto start = std::chrono::high_resolution_clock::now();

  // Use multi_loader for unified interface
  gguf_multi_loader loader;
  if (!loader.load(file_paths)) {
    std::cerr << "Failed to load GGUF file(s)\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "✓ Model loaded successfully in " << duration.count() << " ms\n\n";

  // Test: Basic metadata
  std::cout << "Basic Information:\n";
  std::cout << "  Version: " << loader.get_version() << "\n";
  std::cout << "  Tensor count: " << loader.get_tensor_count() << "\n";
  std::cout << "  Metadata entries: " << loader.get_kv_count() << "\n";
  std::cout << "  Files: " << file_paths.size() << "\n";
  std::cout << "\n";

  // Test: Metadata
  std::cout << "\nAll Metadata:\n";
  auto keys = loader.get_metadata_keys();
  for (const auto& key : keys) {
    std::cout << "  " << key << ": ";
    
    if (auto s = loader.get_string(key)) std::cout << *s;
    else if (auto u32 = loader.get_uint32(key)) std::cout << *u32;
    else if (auto u64 = loader.get_uint64(key)) std::cout << *u64;
    else if (auto f32 = loader.get_float32(key)) std::cout << *f32;
    else if (auto f64 = loader.get_float64(key)) std::cout << *f64;
    else if (auto b = loader.get_bool(key)) std::cout << (*b ? "true" : "false");
    else std::cout << "(array or unknown type)";
    
    std::cout << "\n";
  }

  std::cout << "\n";

  // Test: Attention parameters

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
  }

  auto expert_used_count = loader.get_uint32("gpt-oss.expert_used_count");
  if (expert_used_count) {
    std::cout << "  Experts used (top-k): " << *expert_used_count << "\n";
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

  auto merges = loader.get_array_string("tokenizer.ggml.merges");
  std::cout << "  BPE merges: " << merges.size() << "\n";

  std::cout << "\n";

  // Test: Tensor information
  std::cout << "Tensor Information:\n";

  auto tensor_names = loader.get_tensor_names();
  std::cout << "  Total tensors: " << tensor_names.size() << "\n\n";

  // Show all tensors
  std::cout << "All Tensors:\n";
  for (size_t i = 0; i < tensor_names.size(); ++i) {
    auto info = loader.get_tensor_info(tensor_names[i]);
    if (info) {
      std::cout << tensor_names[i];
      std::cout << " [";
      for (size_t j = 0; j < info->shape.size(); ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << info->shape[j];
      }
      std::cout << "] " << ggml_type_name(info->type);
      std::cout << " size=" << info->size << " bytes\n";
    }
  }

  // Test: Chat template
  std::cout << "\nChat Template:\n";
  auto chat_template = loader.get_string("tokenizer.chat_template");
  if (chat_template) {
    std::cout << "  Template found: " << chat_template->substr(0, 100);
    if (chat_template->size() > 100) {
      std::cout << "... (length: " << chat_template->size() << " chars)";
    }
    std::cout << "\n";
  } else {
    std::cout << "  No chat template found\n";
  }

  std::cout << "\n✓ All tests passed!\n";

  return 0;
}
