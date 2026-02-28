#include <spdlog/spdlog.h>

#include <algorithm>
#include <iostream>

#include "coalsack/gguf/gguf_multi_loader.h"
#include "coalsack/llm/gpt2_tokenizer.h"
#include "coalsack/llm/llm_engine.h"

// Greedy sampler: returns the index of the maximum logit
static uint32_t sample_greedy(const std::vector<float>& logits) {
  return static_cast<uint32_t>(
      std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <gguf_path> [<gguf_path2> ...]\n";
    std::cerr << "\nThis test loads a GGUF model and verifies the minimal graph construction.\n";
    return 1;
  }

  spdlog::set_level(spdlog::level::trace);

  // Collect all GGUF file paths from command line arguments
  std::vector<std::string> gguf_paths;
  for (int i = 1; i < argc; ++i) {
    gguf_paths.push_back(argv[i]);
  }

  std::cout << "=== GPT-OSS Engine Test ===\n\n";

  // Load GGUF
  auto loader = std::make_shared<coalsack::gguf_multi_loader>();
  std::cout << "Loading GGUF from " << gguf_paths.size() << " file(s):\n";
  for (const auto& path : gguf_paths) {
    std::cout << "  - " << path << "\n";
  }
  if (!loader->load(gguf_paths)) {
    std::cerr << "ERROR: Failed to load GGUF file(s)\n";
    return 1;
  }

  // Load tokenizer
  coalsack::gpt2_tokenizer tokenizer;
  if (!tokenizer.load(*loader)) {
    std::cerr << "ERROR: Failed to load tokenizer\n";
    return 1;
  }
  std::cout << "  Vocab size (tokenizer): " << tokenizer.vocab_size() << "\n";
  std::cout << "  BOS: " << tokenizer.bos_token_id() << "  EOS: " << tokenizer.eos_token_id()
            << "\n\n";

  // Load engine
  coalsack::llm_engine::config engine_config;
  engine_config.kv_cache_size = 512;
  engine_config.moe_cache_size_bytes = 536870912;  // 512 MiB per layer
  coalsack::llm_engine engine(engine_config);
  std::cout << "  KV cache size: " << engine_config.kv_cache_size << " tokens\n\n";

  engine.load(loader);

  std::cout << "\n=== Model Info ===\n";
  std::cout << "Loaded: " << (engine.is_loaded() ? "yes" : "no") << "\n";
  std::cout << "Num layers: " << engine.get_num_layers() << "\n";
  std::cout << "Hidden dim: " << engine.get_hidden_dim() << "\n";

  // Encode prompt
  std::string prompt = "Hello";
  std::cout << "\n=== Generation Test ===\n";
  std::cout << "Prompt: \"" << prompt << "\"\n";

  std::vector<uint32_t> prompt_tokens = tokenizer.encode(prompt);
  if (tokenizer.add_bos_token()) {
    const uint32_t bos = tokenizer.bos_token_id();
    if (prompt_tokens.empty() || prompt_tokens.front() != bos) {
      prompt_tokens.insert(prompt_tokens.begin(), bos);
    }
  }

  // Generation loop
  const size_t max_new_tokens = 10;
  const uint32_t eos_id = tokenizer.eos_token_id();
  std::cout << "Generating (max " << max_new_tokens << " tokens, greedy)...\n\n";

  std::vector<uint32_t> output_tokens;
  engine.start(prompt_tokens);
  for (size_t step = 0; step < max_new_tokens; ++step) {
    uint32_t token = sample_greedy(engine.get_logits());
    if (token == eos_id) break;
    output_tokens.push_back(token);
    engine.next(token);
  }
  engine.stop();

  std::string result = tokenizer.decode(output_tokens);

  std::cout << "\n=== Result ===\n";
  std::cout << "Generated: \"" << result << "\"\n";

  std::cout << "\n=== Test Complete ===\n";
  return 0;
}
