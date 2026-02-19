#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "coalsack/llm/chat_template.h"
#include "coalsack/llm/gpt_oss_engine.h"

using namespace coalsack;
using json = nlohmann::json;

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

  // Load gpt_oss_engine
  std::cout << "Loading gpt_oss_engine...\n";

  // model_path is always an array (for both single and multi-file models)
  std::vector<std::string> model_paths = config["model_path"].get<std::vector<std::string>>();

  std::cout << "  Model files (" << model_paths.size() << "):\n";
  for (const auto& path : model_paths) {
    std::cout << "    - " << path << "\n";
  }

  gpt_oss_engine::config engine_config;
  if (config.contains("n_ctx")) {
    engine_config.kv_cache_size = config["n_ctx"];
  } else {
    engine_config.kv_cache_size = 4096;
  }
  engine_config.moe_cache_size_bytes = 1073741824;  // 1 GiB per layer
  std::cout << "  KV cache size: " << engine_config.kv_cache_size << " tokens\n";
  gpt_oss_engine engine(engine_config);

  if (!engine.load(model_paths)) {
    std::cerr << "Failed to load engine\n";
    return 1;
  }
  std::cout << "  Vocab size: " << engine.get_vocab_size() << "\n";
  std::cout << "  Layers: " << engine.get_num_layers() << "\n";
  std::cout << "  Hidden dim: " << engine.get_hidden_dim() << "\n";
  std::cout << "  ✓ Engine loaded successfully\n\n";

  std::cout << "Generating (max " << max_tokens << " tokens, temp=" << temperature << ")...\n";
  std::cout << "\n--- Response ---\n" << std::flush;

  std::string result = engine.generate(prompt, max_tokens, temperature);
  std::cout << result << std::flush;

  std::cout << "\n--- End of Response ---\n\n";
  std::cout << "=================================\n";
  std::cout << "✓ Generation test completed!\n";
  std::cout << "=================================\n";

  return 0;
}
