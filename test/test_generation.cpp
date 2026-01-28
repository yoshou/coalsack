#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "chat_template.h"
#include "llama_backend.h"

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

  const std::string model_path = config["model_path"];
  const int n_ctx = config["n_ctx"];
  const int max_tokens = config["max_tokens"];
  const float temperature = config["temperature"];

  std::vector<int> stop_tokens;
  for (const auto& t : config["stop_tokens"]) {
    stop_tokens.push_back(t.get<int>());
  }

  // Build chat prompt
  std::cout << "Building chat prompt...\n";
  chat_template tpl;
  tpl.add_system(config["system"]);
  tpl.add_user(config["user"]);

  std::string prompt = tpl.build_prompt();
  std::cout << "  Prompt built (" << prompt.size() << " chars)\n";

  // Load llama.cpp backend
  std::cout << "Loading llama.cpp backend...\n";
  llama_backend backend;
  if (!backend.load(model_path, n_ctx)) {
    std::cerr << "Failed to load backend\n";
    return 1;
  }
  std::cout << "  Backend loaded (vocab: " << backend.get_n_vocab()
            << ", ctx: " << backend.get_n_ctx() << ")\n\n";

  // Tokenize using llama.cpp's tokenizer (which handles control tokens correctly)
  std::vector<uint32_t> prompt_tokens = backend.tokenize(prompt, false);
  std::cout << "  Prompt tokenized (" << prompt_tokens.size() << " tokens)\n\n";

  std::cout << "Generating (max " << max_tokens << " tokens, temp=" << temperature << ")...\n";
  std::cout << "Response: " << std::flush;

  std::vector<uint32_t> generated_tokens;
  int n_past = 0;

  // First evaluation: process entire prompt
  std::vector<float> logits = backend.eval(prompt_tokens, n_past);
  if (logits.empty()) {
    std::cerr << "\nERROR: initial eval() failed\n";
    return 1;
  }
  n_past += prompt_tokens.size();

  for (int i = 0; i < max_tokens; ++i) {
    // Greedy sampling: select token with highest logit
    int next_token = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));

    // Check stop tokens
    bool is_stop = false;
    for (int st : stop_tokens) {
      if (next_token == st) {
        is_stop = true;
        break;
      }
    }
    if (is_stop) {
      break;
    }

    generated_tokens.push_back(static_cast<uint32_t>(next_token));

    // Decode and print
    std::vector<uint32_t> single_token = {static_cast<uint32_t>(next_token)};
    std::string piece = backend.detokenize(single_token);
    std::cout << piece << std::flush;

    // Evaluate next token
    logits = backend.eval(single_token, n_past);
    if (logits.empty()) {
      std::cerr << "\nERROR: eval() failed at token " << i << "\n";
      return 1;
    }
    n_past += 1;
  }

  std::cout << "\n\n✓ Generation test completed!\n";

  return 0;
}
