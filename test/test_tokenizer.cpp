#include <chrono>
#include <iostream>

#include "gguf_loader.h"
#include "gpt2_tokenizer.h"

using namespace coalsack;

int main(int argc, char** argv) {
  std::cout << "Testing GPT2 Tokenizer\n";
  std::cout << "======================\n\n";

  std::string model_path = "/workspaces/stargazer/models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  gguf_loader loader;
  if (!loader.load(model_path)) {
    std::cerr << "Failed to load GGUF file\n";
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();

  gpt2_tokenizer tokenizer;
  if (!tokenizer.load_from_gguf(loader)) {
    std::cerr << "Failed to load tokenizer\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "✓ Tokenizer loaded in " << duration.count() << " ms\n\n";

  std::cout << "Tokenizer Info:\n";
  std::cout << "  Vocab size: " << tokenizer.vocab_size() << "\n";
  std::cout << "  BOS token ID: " << tokenizer.bos_token_id() << "\n";
  std::cout << "  EOS token ID: " << tokenizer.eos_token_id() << "\n\n";

  if (tokenizer.vocab_size() != 201088) {
    std::cerr << "ERROR: Expected 201088 vocab size, got " << tokenizer.vocab_size() << "\n";
    return 1;
  }

  // Test 1: English text
  std::cout << "Test 1: English text\n";
  std::string text1 = "Hello, world!";
  auto tokens1 = tokenizer.encode(text1);
  std::cout << "  Input: \"" << text1 << "\"\n";
  std::cout << "  Tokens: [";
  for (size_t i = 0; i < tokens1.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << tokens1[i];
  }
  std::cout << "]\n";

  std::vector<uint32_t> expected1 = {13225, 11, 2375, 0};
  if (tokens1 == expected1) {
    std::cout << "  ✓ Tokens match expected\n";
  } else {
    std::cerr << "  ERROR: Expected [13225, 11, 2375, 0]\n";
    return 1;
  }

  auto decoded1 = tokenizer.decode(tokens1);
  std::cout << "  Decoded: \"" << decoded1 << "\"\n";
  if (decoded1 == text1) {
    std::cout << "  ✓ Round-trip successful\n";
  } else {
    std::cerr << "  ERROR: Decoded text doesn't match\n";
    return 1;
  }
  std::cout << "\n";

  // Test 2: Japanese text
  std::cout << "Test 2: Japanese text\n";
  std::string text2 = "こんにちは！";
  auto tokens2 = tokenizer.encode(text2);
  std::cout << "  Input: \"" << text2 << "\"\n";
  std::cout << "  Tokens: [";
  for (size_t i = 0; i < tokens2.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << tokens2[i];
  }
  std::cout << "]\n";

  std::vector<uint32_t> expected2 = {95839, 3393};
  if (tokens2 == expected2) {
    std::cout << "  ✓ Tokens match expected\n";
  } else {
    std::cerr << "  ERROR: Expected [95839, 3393]\n";
    return 1;
  }

  auto decoded2 = tokenizer.decode(tokens2);
  std::cout << "  Decoded: \"" << decoded2 << "\"\n";
  if (decoded2 == text2) {
    std::cout << "  ✓ Round-trip successful\n";
  } else {
    std::cerr << "  ERROR: Decoded text doesn't match\n";
    return 1;
  }

  std::cout << "\n✓ All tests passed!\n";

  return 0;
}
