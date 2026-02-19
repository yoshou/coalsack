#include <spdlog/spdlog.h>

#include <iostream>

#include "coalsack/llm/gpt_oss_engine.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <gguf_path> [<gguf_path2> ...]\n";
    std::cerr << "\nThis test loads a GGUF model and verifies the minimal graph construction.\n";
    std::cerr << "Note: Only float32 GGUF files are supported currently.\n";
    return 1;
  }

  spdlog::set_level(spdlog::level::trace);

  // Collect all GGUF file paths from command line arguments
  std::vector<std::string> gguf_paths;
  for (int i = 1; i < argc; ++i) {
    gguf_paths.push_back(argv[i]);
  }

  std::cout << "=== GPT-OSS Engine Test ===\n\n";

  coalsack::gpt_oss_engine::config engine_config;
  engine_config.kv_cache_size = 4096;
  engine_config.moe_cache_size_bytes = 1073741824;  // 1 GiB per layer
  coalsack::gpt_oss_engine engine(engine_config);

  std::cout << "Loading model from " << gguf_paths.size() << " file(s):\n";
  for (const auto& path : gguf_paths) {
    std::cout << "  - " << path << "\n";
  }
  std::cout << "  KV cache size: " << engine_config.kv_cache_size << " tokens\n\n";

  if (!engine.load(gguf_paths)) {
    std::cerr << "ERROR: Failed to load model\n";
    return 1;
  }

  std::cout << "\n=== Model Info ===\n";
  std::cout << "Loaded: " << (engine.is_loaded() ? "yes" : "no") << "\n";
  std::cout << "Vocab size: " << engine.get_vocab_size() << "\n";
  std::cout << "Num layers: " << engine.get_num_layers() << "\n";
  std::cout << "Hidden dim: " << engine.get_hidden_dim() << "\n";

  std::cout << "\n=== Generation Test ===\n";
  std::string prompt = "Hello";
  std::cout << "Prompt: \"" << prompt << "\"\n";
  std::cout << "Generating (max 10 tokens, temperature 0.0)...\n\n";

  std::string result = engine.generate(prompt, 10, 0.0f);

  std::cout << "\n=== Result ===\n";
  std::cout << "Generated: \"" << result << "\"\n";

  std::cout << "\n=== Test Complete ===\n";
  return 0;
}
