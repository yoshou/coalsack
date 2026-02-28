#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "coalsack/gguf/gguf_multi_loader.h"
#include "coalsack/llm/chat_template.h"
#include "coalsack/llm/gpt2_tokenizer.h"
#include "coalsack/llm/llm_engine.h"

using namespace coalsack;
using json = nlohmann::json;

// Sample next token from logits with temperature.
// temperature <= 0 uses greedy (argmax).
static uint32_t sample_token(const std::vector<float>& logits, float temperature) {
  const int64_t vocab_size = static_cast<int64_t>(logits.size());
  if (temperature < 1e-6f) {
    return static_cast<uint32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  std::vector<float> probs(vocab_size);
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) / temperature);
    sum += probs[i];
  }
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] /= sum;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return static_cast<uint32_t>(dist(gen));
}

int main(int argc, char** argv) {
  // Set log level from environment variable
  const char* log_level_env = std::getenv("LOG_LEVEL");
  if (log_level_env) {
    std::string level_str(log_level_env);
    if (level_str == "trace") {
      spdlog::set_level(spdlog::level::trace);
    } else if (level_str == "debug") {
      spdlog::set_level(spdlog::level::debug);
    } else if (level_str == "info") {
      spdlog::set_level(spdlog::level::info);
    } else if (level_str == "warn") {
      spdlog::set_level(spdlog::level::warn);
    } else if (level_str == "error") {
      spdlog::set_level(spdlog::level::err);
    }
  }

  std::cout << "=================================\n";
  std::cout << "GPT-OSS Generation Test\n";
  std::cout << "=================================\n\n";

  std::string json_path = argc > 1 ? argv[1] : "test/generation_test_data.json";

  std::ifstream json_file(json_path);
  if (!json_file) {
    std::cerr << "ERROR: Cannot open " << json_path << "\n";
    return 1;
  }

  json config;
  try {
    json_file >> config;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: Failed to parse JSON: " << e.what() << "\n";
    return 1;
  }

  const int max_tokens = config["max_tokens"];
  const float temperature = config["temperature"];

  // Build chat prompt
  std::cout << "Building chat prompt...\n";

  if (!config.contains("messages")) {
    std::cerr << "ERROR: 'messages' field is required in JSON config\n";
    return 1;
  }

  chat_template tpl;
  for (const auto& msg : config["messages"]) {
    if (!msg.contains("role") || !msg.contains("content")) {
      std::cerr << "ERROR: Each message must have 'role' and 'content' fields\n";
      return 1;
    }
    std::string role = msg["role"];
    std::string content = msg["content"];
    if (role == "system") {
      tpl.add_system(content);
    } else if (role == "user") {
      tpl.add_user(content);
    } else if (role == "assistant") {
      tpl.add_assistant(content);
    } else {
      std::cerr << "WARNING: Unknown role '" << role << "' (skipped)\n";
    }
  }

  std::string prompt = tpl.build_prompt();
  std::cout << "  ✓ Prompt built (" << prompt.size() << " chars)\n";
  std::cout << "\n--- Prompt ---\n" << prompt << "\n--- End of Prompt ---\n\n";

  // model_path is always an array (for both single and multi-file models)
  std::vector<std::string> model_paths = config["model_path"].get<std::vector<std::string>>();

  std::cout << "Loading GGUF (" << model_paths.size() << " file(s)):\n";
  for (const auto& path : model_paths) {
    std::cout << "    - " << path << "\n";
  }

  auto loader = std::make_shared<gguf_multi_loader>();
  if (!loader->load(model_paths)) {
    std::cerr << "ERROR: Failed to load GGUF file(s)\n";
    return 1;
  }

  // Load tokenizer
  gpt2_tokenizer tokenizer;
  if (!tokenizer.load(*loader)) {
    std::cerr << "ERROR: Failed to load tokenizer\n";
    return 1;
  }
  const uint32_t eos_id = tokenizer.eos_token_id();
  std::cout << "  Vocab size (tokenizer): " << tokenizer.vocab_size() << "  EOS: " << eos_id
            << "\n";

  // Load engine
  llm_engine::config engine_config;
  if (config.contains("n_ctx")) {
    engine_config.kv_cache_size = config["n_ctx"];
  } else {
    engine_config.kv_cache_size = 4096;
  }
  engine_config.moe_cache_size_bytes = 1073741824;  // 1 GiB per layer
  std::cout << "  KV cache size: " << engine_config.kv_cache_size << " tokens\n";

  llm_engine engine(engine_config);
  engine.load(loader);

  std::cout << "  Layers: " << engine.get_num_layers() << "\n";
  std::cout << "  Hidden dim: " << engine.get_hidden_dim() << "\n";
  std::cout << "  ✓ Engine loaded successfully\n\n";

  // Encode prompt
  std::vector<uint32_t> prompt_tokens = tokenizer.encode(prompt);
  if (tokenizer.add_bos_token()) {
    const uint32_t bos = tokenizer.bos_token_id();
    if (prompt_tokens.empty() || prompt_tokens.front() != bos) {
      prompt_tokens.insert(prompt_tokens.begin(), bos);
    }
  }

  std::cout << "Generating (max " << max_tokens << " tokens, temp=" << temperature << ")...\n";
  std::cout << "\n--- Response ---\n" << std::flush;

  // Generation loop
  std::vector<uint32_t> output_tokens;
  auto logits = engine.start(prompt_tokens);
  for (int step = 0; step < max_tokens; ++step) {
    uint32_t token = sample_token(logits, temperature);
    if (token == eos_id) break;
    output_tokens.push_back(token);
    std::cout << tokenizer.decode({token}) << std::flush;
    logits = engine.next(token);
  }
  engine.stop();

  std::cout << "\n--- End of Response ---\n\n";
  std::cout << "=================================\n";
  std::cout << "✓ Generation test completed!\n";
  std::cout << "=================================\n";

  return 0;
}
