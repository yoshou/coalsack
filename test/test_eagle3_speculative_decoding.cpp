#include <spdlog/spdlog.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/gguf/gguf_multi_loader.h"
#include "coalsack/llm/eagle3_speculative_decoder.h"
#include "coalsack/llm/gpt2_tokenizer.h"
#include "coalsack/llm/llm_engine.h"
#include "coalsack/llm/sampler.h"
#include "coalsack/tensor/dynamic_tensor.h"

using namespace coalsack;

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
  spdlog::set_level(spdlog::level::info);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <target_gguf> <eagle3_gguf>\n";
    return 1;
  }

  eagle3_speculative_decoder::config eagle3_cfg;
  eagle3_cfg.max_seq_len = 512;
  eagle3_speculative_decoder eagle3(eagle3_cfg);
  if (!eagle3.load(argv[2])) {
    std::cerr << "FAIL: Eagle3 load failed\n";
    return 1;
  }

  const auto& extract_layers = eagle3.get_extract_layers();

  auto loader = std::make_shared<gguf_multi_loader>();
  if (!loader->load({argv[1]})) {
    std::cerr << "FAIL: Target GGUF load failed\n";
    return 1;
  }

  gpt2_tokenizer tokenizer;
  if (!tokenizer.load(*loader)) {
    std::cerr << "FAIL: Tokenizer load failed\n";
    return 1;
  }

  llm_engine::config engine_cfg;
  engine_cfg.kv_cache_size = 512;
  engine_cfg.moe_cache_size_bytes = 1073741824;
  for (int l : extract_layers) engine_cfg.hidden_layer_indices.push_back(l - 1);

  llm_engine target(engine_cfg);
  target.load(loader);

  // Prompt: "The capital of France is Paris. The Eiffel Tower was built in"
  // (chat-template applied, 80 tokens)
  const std::vector<uint32_t> prompt_tokens = {
      200006, 17360,  200008, 3575,   553,   17554,  162016, 11,    261,   4410,  6439, 2359,
      22203,  656,    7788,   17527,  558,   87447,  100594, 25,    220,   1323,  19,   12,
      3218,   198,    6576,   3521,   25,    220,    1323,   21,    12,    3659,  12,   2290,
      279,    30377,  289,    25,     14093, 279,    2,      13888, 18403, 25,    8450, 11,
      49159,  11,     1721,   13,     21030, 2804,   413,    7360,  395,   1753,  3176, 13,
      200007, 200006, 1428,   200008, 976,   9029,   328,    10128, 382,   12650, 13,   623,
      155511, 37994,  673,    8113,   306,   200007, 200006, 173781};

  const int64_t n_prompt = static_cast<int64_t>(prompt_tokens.size());

  target.start(prompt_tokens);

  sampler target_sampler;  // greedy (temperature=0)
  sampler draft_sampler;   // greedy (temperature=0)

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

  // === Speculative decode loop ===
  const int N_DRAFT = 8;
  const float p_min = 0.0f;
  const int max_new_tokens = 50;

  std::cout << tokenizer.decode({prompt_last_token});
  std::cout.flush();

  int64_t eagle3_pos = n_prompt;
  int64_t target_pos = n_prompt;
  uint32_t cur_token = prompt_last_token;

  std::vector<float> cur_draft_logits(
      eagle3.get_logits().data_ptr<float>() + (n_prompt - 1) * draft_vocab,
      eagle3.get_logits().data_ptr<float>() + (n_prompt - 1) * draft_vocab + draft_vocab);

  int total_draft = 0;
  int total_accepted = 0;
  int total_tokens = 0;

  while (true) {
    // 1. Generate up to N_DRAFT tokens (d0 always generated; p_min applies to d1+)
    std::vector<int64_t> draft_ids;
    dynamic_tensor prenorm = eagle3.get_prenorm();

    {
      draft_ids.push_back(
          sample_draft(draft_sampler, eagle3, cur_draft_logits.data(), draft_vocab, target_vocab));
      float p0 = draft_sampler.get_top1_prob_after_topk(cur_draft_logits.data(), draft_vocab);
      if (p0 < p_min) goto draft_done;
    }

    for (int j = 1; j < N_DRAFT && static_cast<int>(draft_ids.size()) == j; ++j) {
      int64_t last_idx = prenorm.dim(1) - 1;
      size_t offset =
          static_cast<size_t>(last_idx) * static_cast<size_t>(eagle3_hidden_size) * sizeof(float);
      dynamic_tensor step_g_embd = prenorm.make_view({1, 1, eagle3_hidden_size}, offset);
      eagle3.decode({static_cast<uint32_t>(draft_ids.back())}, step_g_embd, eagle3_pos + j - 1);
      const auto& step_logits = eagle3.get_logits();
      float pj = draft_sampler.get_top1_prob_after_topk(step_logits.data_ptr<float>(), draft_vocab);
      if (pj < p_min) break;
      draft_ids.push_back(sample_draft(draft_sampler, eagle3, step_logits.data_ptr<float>(),
                                       draft_vocab, target_vocab));
      prenorm = eagle3.get_prenorm();
    }
  draft_done:

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
    total_tokens += n_new;

    for (int j = 0; j < result.n_accepted; ++j) {
      std::cout << tokenizer.decode({static_cast<uint32_t>(draft_ids[j])});
      std::cout.flush();
    }
    std::cout << tokenizer.decode({result.correction_token});
    std::cout.flush();

    cur_token = result.correction_token;

    if (total_tokens >= max_new_tokens) break;

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

    std::vector<uint32_t> committed_tokens;
    for (int j = 0; j < result.n_accepted; ++j)
      committed_tokens.push_back(static_cast<uint32_t>(draft_ids[j]));
    committed_tokens.push_back(result.correction_token);
    eagle3.decode(committed_tokens, next_g_embd, eagle3_pos);
    eagle3_pos += n_new;

    // 8. Update logits for next d0
    const float* logits_ptr = eagle3.get_logits().data_ptr<float>() + (n_new - 1) * draft_vocab;
    cur_draft_logits.assign(logits_ptr, logits_ptr + draft_vocab);
  }

  target.stop();
  eagle3.stop();

  std::cout << "\n";
  const double rate = total_draft > 0
                          ? static_cast<double>(total_accepted) / static_cast<double>(total_draft)
                          : 0.0;
  std::cout << "Acceptance: " << total_accepted << "/" << total_draft << " = "
            << static_cast<int>(rate * 100.0 + 0.5) << "%\n";
  std::cout << "Total new tokens: " << (total_tokens + 1) << " (incl. prompt_last_token)\n";

  return 0;
}
