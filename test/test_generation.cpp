#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "chat_template.h"
#include "gpt_oss_engine.h"

using namespace coalsack;
using json = nlohmann::json;

int main(int argc, char** argv) {
  std::cout << "Testing gpt-oss-20b Generation\n";
  std::cout << "===============================\n\n";

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
  chat_template tpl;
  tpl.add_system(config["system"]);
  tpl.add_user(config["user"]);

  std::string prompt = tpl.build_prompt();
  std::cout << "  Prompt built (" << prompt.size() << " chars)\n\n";

  // Load gpt_oss_engine
  std::cout << "Loading gpt_oss_engine...\n";
  
  // model_path is always an array (for both single and multi-file models)
  std::vector<std::string> model_paths = config["model_path"].get<std::vector<std::string>>();
  
  std::cout << "  Model files (" << model_paths.size() << "):\n";
  for (const auto& path : model_paths) {
    std::cout << "    - " << path << "\n";
  }
  
  gpt_oss_engine::config engine_config;
  engine_config.kv_cache_size = 4096;
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
  std::cout << "  Engine loaded successfully\n\n";

  std::cout << "Generating (max " << max_tokens << " tokens, temp=" << temperature << ")...\n";
  std::cout << "Response: " << std::flush;

  std::string result = engine.generate(prompt, max_tokens, temperature);
  std::cout << result << std::flush;

  std::cout << "\n\n✓ Generation test completed!\n";

  return 0;
}
