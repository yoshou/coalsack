#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/gguf/gguf_multi_loader.h"
#include "coalsack/llm/chat_template.h"
#include "coalsack/llm/eagle3_speculative_decoder.h"
#include "coalsack/llm/gpt2_tokenizer.h"
#include "coalsack/llm/llm_engine.h"
#include "coalsack/llm/sampler.h"
#include "coalsack/tensor/dynamic_tensor.h"

using namespace coalsack;
using json = nlohmann::json;

static void set_log_level_from_env() {
  const char* log_level_env = std::getenv("LOG_LEVEL");
  if (!log_level_env) return;

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

static std::string build_prompt_from_messages(const json& config) {
  if (!config.contains("messages")) {
    throw std::runtime_error("'messages' field is required in JSON config");
  }

  chat_template tpl;
  for (const auto& msg : config["messages"]) {
    if (!msg.contains("role") || !msg.contains("content")) {
      throw std::runtime_error("Each message must have 'role' and 'content' fields");
    }

    const std::string role = msg["role"];
    const std::string content = msg["content"];
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

  return tpl.build_prompt();
}

static std::string read_draft_model_path(const json& config) {
  if (config.contains("draft_model_path")) {
    return config["draft_model_path"].get<std::string>();
  }
  if (config.contains("eagle3_model_path")) {
    return config["eagle3_model_path"].get<std::string>();
  }
  throw std::runtime_error("'draft_model_path' field is required in JSON config");
}

// Sample a token in draft-vocab space and map it to target-vocab space.
static uint32_t sample_draft(sampler& smp, const eagle3_speculative_decoder& eagle3,
                             const float* draft_logits, int64_t draft_vocab, int64_t target_vocab) {
  const uint32_t draft_tok = smp.sample(draft_logits, draft_vocab);
  const int64_t tgt = eagle3.draft_to_target(static_cast<int64_t>(draft_tok));
  if (tgt >= 0 && tgt < target_vocab) return static_cast<uint32_t>(tgt);
  throw std::runtime_error("sample_draft: draft_to_target returned out-of-range id " +
                           std::to_string(tgt) + " for draft token " + std::to_string(draft_tok));
}

int main(int argc, char** argv) {
  set_log_level_from_env();

  std::cout << "=================================\n";
  std::cout << "GPT-OSS Eagle3 Speculative Decoding Test\n";
  std::cout << "=================================\n\n";

  const std::string json_path =
      argc > 1 ? argv[1] : "test/eagle3_speculative_decoding_test_data.json";

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

  std::vector<std::string> model_paths;
  std::string draft_model_path;
  std::string prompt;
  int64_t n_ctx = 512;
  int max_tokens = 50;
  float temperature = 0.0f;
  int n_draft_max = 8;
  float p_min = 0.0f;

  try {
    if (!config.contains("model_path")) {
      throw std::runtime_error("'model_path' field is required in JSON config");
    }
    model_paths = config["model_path"].get<std::vector<std::string>>();
    draft_model_path = read_draft_model_path(config);
    prompt = build_prompt_from_messages(config);
    n_ctx = config.value("n_ctx", 512);
    max_tokens = config.value("max_tokens", 50);
    temperature = config.value("temperature", 0.0f);
    n_draft_max = config.value("n_draft", 8);
    p_min = config.value("p_min", 0.0f);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  if (model_paths.empty()) {
    std::cerr << "ERROR: 'model_path' must contain at least one file\n";
    return 1;
  }
  if (n_ctx <= 0) {
    std::cerr << "ERROR: 'n_ctx' must be positive\n";
    return 1;
  }
  if (n_draft_max <= 0) {
    std::cerr << "ERROR: 'n_draft' must be positive\n";
    return 1;
  }

  std::cout << "Building chat prompt...\n";
  std::cout << "  ✓ Prompt built (" << prompt.size() << " chars)\n";
  std::cout << "\n--- Prompt ---\n" << prompt << "\n--- End of Prompt ---\n\n";

  eagle3_speculative_decoder::config eagle3_cfg;
  eagle3_cfg.max_seq_len = n_ctx;
  eagle3_speculative_decoder eagle3(eagle3_cfg);

  std::cout << "Loading GGUF (" << model_paths.size() << " file(s)):\n";
  for (const auto& path : model_paths) {
    std::cout << "    - " << path << "\n";
  }
  std::cout << "Loading Eagle3 GGUF:\n";
  std::cout << "    - " << draft_model_path << "\n";

  if (!eagle3.load(draft_model_path)) {
    std::cerr << "ERROR: Eagle3 load failed\n";
    return 1;
  }

  const auto& extract_layers = eagle3.get_extract_layers();
  if (extract_layers.empty()) {
    std::cerr << "ERROR: Eagle3 did not provide extract layers\n";
    return 1;
  }

  auto loader = std::make_shared<gguf_multi_loader>();
  if (!loader->load(model_paths)) {
    std::cerr << "ERROR: Failed to load GGUF file(s)\n";
    return 1;
  }

  gpt2_tokenizer tokenizer;
  if (!tokenizer.load(*loader)) {
    std::cerr << "ERROR: Failed to load tokenizer\n";
    return 1;
  }
  const uint32_t eos_id = tokenizer.eos_token_id();
  std::cout << "  Vocab size (tokenizer): " << tokenizer.vocab_size() << "  EOS: " << eos_id
            << "\n";

  llm_engine::config engine_cfg;
  engine_cfg.kv_cache_size = n_ctx;
  engine_cfg.moe_cache_size_bytes = 1073741824;
  for (int l : extract_layers) engine_cfg.hidden_layer_indices.push_back(l - 1);
  std::cout << "  KV cache size: " << engine_cfg.kv_cache_size << " tokens\n";

  llm_engine target(engine_cfg);
  target.load(loader);

  std::cout << "  Layers: " << target.get_num_layers() << "\n";
  std::cout << "  Hidden dim: " << target.get_hidden_dim() << "\n";
  std::cout << "  Draft width: " << n_draft_max << "  p_min: " << p_min << "  temp=" << temperature
            << "\n";
  std::cout << "  ✓ Engine loaded successfully\n\n";

  std::vector<uint32_t> prompt_tokens = tokenizer.encode(prompt);
  if (tokenizer.add_bos_token()) {
    const uint32_t bos = tokenizer.bos_token_id();
    if (prompt_tokens.empty() || prompt_tokens.front() != bos) {
      prompt_tokens.insert(prompt_tokens.begin(), bos);
    }
  }

  if (prompt_tokens.empty()) {
    std::cerr << "ERROR: Prompt tokenization produced no tokens\n";
    return 1;
  }

  const int64_t n_prompt = static_cast<int64_t>(prompt_tokens.size());

  target.start(prompt_tokens);

  sampler::config sampler_cfg;
  sampler_cfg.temperature = temperature;
  sampler target_sampler(sampler_cfg);
  sampler draft_sampler(sampler_cfg);

  const auto& prompt_logits = target.get_logits();
  const uint32_t prompt_last_token =
      target_sampler.sample(prompt_logits.data(), static_cast<int64_t>(prompt_logits.size()));

  std::unordered_map<int, std::vector<float>> prompt_hidden;
  for (int l : extract_layers) prompt_hidden[l] = target.get_hidden_layer_all_pos(l - 1);

  const int64_t target_hidden_size =
      static_cast<int64_t>(prompt_hidden.begin()->second.size()) / n_prompt;

  dynamic_tensor g_embd = eagle3.encode(prompt_hidden, n_prompt);

  std::vector<uint32_t> eagle3_prefill_tokens(n_prompt);
  for (int64_t i = 0; i < n_prompt - 1; ++i) eagle3_prefill_tokens[i] = prompt_tokens[i + 1];
  eagle3_prefill_tokens[n_prompt - 1] = prompt_last_token;

  eagle3.start();
  eagle3.decode(eagle3_prefill_tokens, g_embd, 0);

  const int64_t eagle3_hidden_size = g_embd.dim(2);
  const int64_t draft_vocab = eagle3.get_draft_vocab_size();
  const int64_t target_vocab = static_cast<int64_t>(prompt_logits.size());

  std::cout << "Generating (max " << max_tokens << " tokens, temp=" << temperature << ")...\n";
  std::cout << "\n--- Response ---\n" << std::flush;

  int64_t eagle3_pos = n_prompt;
  int64_t target_pos = n_prompt;
  uint32_t cur_token = prompt_last_token;

  std::vector<float> cur_draft_logits(
      eagle3.get_logits().data_ptr<float>() + (n_prompt - 1) * draft_vocab,
      eagle3.get_logits().data_ptr<float>() + (n_prompt - 1) * draft_vocab + draft_vocab);

  int total_draft = 0;
  int total_accepted = 0;
  int total_generated_tokens = 0;
  bool stop_generation = false;

  if (max_tokens > 0 && prompt_last_token != eos_id) {
    std::cout << tokenizer.decode({prompt_last_token}) << std::flush;
    total_generated_tokens = 1;
  } else {
    stop_generation = true;
  }

  while (!stop_generation) {
    if (total_generated_tokens >= max_tokens) break;

    // 1. Generate up to N_DRAFT tokens (d0 always generated; p_min applies to d1+)
    std::vector<int64_t> draft_ids;
    dynamic_tensor prenorm = eagle3.get_prenorm();

    {
      draft_ids.push_back(
          sample_draft(draft_sampler, eagle3, cur_draft_logits.data(), draft_vocab, target_vocab));
      float p0 = draft_sampler.get_top1_prob_after_topk(cur_draft_logits.data(), draft_vocab);
      if (p0 >= p_min) {
        for (int j = 1; j < n_draft_max && static_cast<int>(draft_ids.size()) == j; ++j) {
          int64_t last_idx = prenorm.dim(1) - 1;
          size_t offset = static_cast<size_t>(last_idx) * static_cast<size_t>(eagle3_hidden_size) *
                          sizeof(float);
          dynamic_tensor step_g_embd = prenorm.make_view({1, 1, eagle3_hidden_size}, offset);
          eagle3.decode({static_cast<uint32_t>(draft_ids.back())}, step_g_embd, eagle3_pos + j - 1);
          const auto& step_logits = eagle3.get_logits();
          float pj =
              draft_sampler.get_top1_prob_after_topk(step_logits.data_ptr<float>(), draft_vocab);
          if (pj < p_min) break;
          draft_ids.push_back(sample_draft(draft_sampler, eagle3, step_logits.data_ptr<float>(),
                                           draft_vocab, target_vocab));
          prenorm = eagle3.get_prenorm();
        }
      }
    }

    const int n_draft = static_cast<int>(draft_ids.size());
    if (n_draft == 0) break;
    total_draft += n_draft;

    // 2. Verify: feed [cur_token, d0, ..., d(n-1)] to target (n_draft+1 tokens)
    std::vector<uint32_t> verify_batch;
    verify_batch.push_back(cur_token);
    for (int j = 0; j < n_draft; ++j) verify_batch.push_back(static_cast<uint32_t>(draft_ids[j]));
    target.next_batch(verify_batch);

    const std::vector<float> verify_logits = target.get_logits_all_pos();

    // 3. Save hidden states for eagle3 re-encode
    std::unordered_map<int, std::vector<float>> verify_hidden;
    for (int l : extract_layers) verify_hidden[l] = target.get_hidden_layer_all_pos(l - 1);

    // 4. Speculative verify
    std::vector<uint32_t> draft_u32(draft_ids.begin(), draft_ids.end());
    const sampler::verify_result result = sampler::speculative_verify(
        draft_u32.data(), verify_logits.data(), n_draft, target_vocab, target_sampler);

    // 5. Commit accepted tokens + correction
    const int n_new = result.n_accepted + 1;
    total_accepted += result.n_accepted;

    std::vector<uint32_t> committed_tokens;
    committed_tokens.reserve(static_cast<size_t>(n_new));
    for (int j = 0; j < result.n_accepted; ++j) {
      committed_tokens.push_back(static_cast<uint32_t>(draft_ids[j]));
    }
    committed_tokens.push_back(result.correction_token);

    int eos_index = -1;
    for (int i = 0; i < static_cast<int>(committed_tokens.size()); ++i) {
      if (committed_tokens[static_cast<size_t>(i)] == eos_id) {
        eos_index = i;
        break;
      }
    }

    const int printable_tokens =
        eos_index >= 0 ? eos_index : static_cast<int>(committed_tokens.size());
    const int remaining_budget = max_tokens - total_generated_tokens;
    const int emitted_tokens = std::min(printable_tokens, remaining_budget);

    for (int i = 0; i < emitted_tokens; ++i) {
      std::cout << tokenizer.decode({committed_tokens[static_cast<size_t>(i)]}) << std::flush;
    }

    total_generated_tokens += emitted_tokens;

    if (eos_index >= 0 || total_generated_tokens >= max_tokens) {
      break;
    }

    cur_token = result.correction_token;

    // 6. Rollback target KV cache to committed length
    target.rollback_to(target_pos + n_new);
    target_pos += n_new;

    // 7. Eagle3: rollback → re-encode → batch decode
    std::unordered_map<int, std::vector<float>> committed_hidden;
    for (int l : extract_layers) {
      const auto& src = verify_hidden.at(l);
      committed_hidden[l] =
          std::vector<float>(src.begin(), src.begin() + n_new * target_hidden_size);
    }

    eagle3.rollback_to(eagle3_pos);
    dynamic_tensor next_g_embd = eagle3.encode(committed_hidden, n_new);

    eagle3.decode(committed_tokens, next_g_embd, eagle3_pos);
    eagle3_pos += n_new;

    // 8. Update logits for next d0
    const float* logits_ptr = eagle3.get_logits().data_ptr<float>() + (n_new - 1) * draft_vocab;
    cur_draft_logits.assign(logits_ptr, logits_ptr + draft_vocab);
  }

  target.stop();
  eagle3.stop();

  std::cout << "\n--- End of Response ---\n\n";
  const double rate = total_draft > 0
                          ? static_cast<double>(total_accepted) / static_cast<double>(total_draft)
                          : 0.0;
  std::cout << "=================================\n";
  std::cout << "✓ Speculative decoding test completed!\n";
  std::cout << "=================================\n";
  std::cout << "Total generated tokens: " << total_generated_tokens << "\n";
  std::cout << "Acceptance: " << total_accepted << "/" << total_draft << " = "
            << static_cast<int>(rate * 100.0 + 0.5) << "%\n";

  return 0;
}
