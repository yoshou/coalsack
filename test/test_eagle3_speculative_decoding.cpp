#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/gguf/gguf_multi_loader.h"
#include "coalsack/llm/eagle3_speculative_decoder.h"
#include "coalsack/llm/gpt2_tokenizer.h"
#include "coalsack/llm/llm_engine.h"
#include "coalsack/tensor/dynamic_tensor.h"

using namespace coalsack;

static uint32_t argmax_f32(const float* data, int64_t size) {
  int64_t best = 0;
  for (int64_t i = 1; i < size; ++i) {
    if (data[i] > data[best]) best = i;
  }
  return static_cast<uint32_t>(best);
}

static uint32_t eagle3_argmax_target_from_draft_logits(const eagle3_speculative_decoder& eagle3,
                                                       const float* draft_logits,
                                                       int64_t draft_vocab, int64_t target_vocab) {
  std::vector<float> target_logits(static_cast<size_t>(target_vocab),
                                   -std::numeric_limits<float>::infinity());
  for (int64_t j = 0; j < draft_vocab; ++j) {
    const int64_t tgt = eagle3.draft_to_target(j);
    if (tgt >= 0 && tgt < target_vocab) {
      target_logits[static_cast<size_t>(tgt)] = draft_logits[j];
    }
  }
  return argmax_f32(target_logits.data(), target_vocab);
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <target_gguf> <eagle3_gguf>\n";
    return 1;
  }

  const std::string target_gguf_path = argv[1];
  const std::string eagle3_gguf_path = argv[2];

  eagle3_speculative_decoder::config eagle3_cfg;
  eagle3_cfg.max_seq_len = 512;

  eagle3_speculative_decoder eagle3(eagle3_cfg);
  if (!eagle3.load(eagle3_gguf_path)) {
    std::cerr << "FAIL: Eagle3 load failed\n";
    return 1;
  }

  const auto& extract_layers = eagle3.get_extract_layers();

  auto loader = std::make_shared<gguf_multi_loader>();
  if (!loader->load({target_gguf_path})) {
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
  for (int l : extract_layers) {
    engine_cfg.hidden_layer_indices.push_back(l - 1);
  }

  llm_engine target(engine_cfg);
  target.load(loader);

  const std::string prompt_text = "The capital of France is Paris. The Eiffel Tower was built in";

  std::vector<uint32_t> prompt_tokens = {
      200006, 17360,  200008, 3575,   553,   17554,  162016, 11,    261,   4410,  6439, 2359,
      22203,  656,    7788,   17527,  558,   87447,  100594, 25,    220,   1323,  19,   12,
      3218,   198,    6576,   3521,   25,    220,    1323,   21,    12,    3659,  12,   2290,
      279,    30377,  289,    25,     14093, 279,    2,      13888, 18403, 25,    8450, 11,
      49159,  11,     1721,   13,     21030, 2804,   413,    7360,  395,   1753,  3176, 13,
      200007, 200006, 1428,   200008, 976,   9029,   328,    10128, 382,   12650, 13,   623,
      155511, 37994,  673,    8113,   306,   200007, 200006, 173781};

  const int64_t n_prompt = static_cast<int64_t>(prompt_tokens.size());

  if (n_prompt < 2) {
    std::cerr << "FAIL: Prompt is too short (need at least 2 tokens)\n";
    return 1;
  }

  target.start(prompt_tokens);

  const auto& tgt_logits_after_prompt = target.get_logits();
  const uint32_t id_last = argmax_f32(tgt_logits_after_prompt.data(),
                                      static_cast<int64_t>(tgt_logits_after_prompt.size()));

  std::unordered_map<int, std::vector<float>> all_hidden;
  for (int l : extract_layers) {
    all_hidden[l] = target.get_hidden_layer_all_pos(l - 1);
  }

  dynamic_tensor g_embd = eagle3.encode(all_hidden, n_prompt);

  std::vector<uint32_t> dec_tokens(n_prompt);
  for (int64_t i = 0; i < n_prompt - 1; ++i) {
    dec_tokens[i] = prompt_tokens[i + 1];
  }
  dec_tokens[n_prompt - 1] = id_last;

  eagle3.start();
  eagle3.decode(dec_tokens, g_embd, 0);

  const int64_t eagle3_H = g_embd.dim(2);
  const int64_t draft_vocab = eagle3.get_draft_vocab_size();
  const int64_t target_vocab = static_cast<int64_t>(tgt_logits_after_prompt.size());

  const auto& prefill_logits = eagle3.get_logits();
  const float* last_pos_logits = prefill_logits.data_ptr<float>() + (n_prompt - 1) * draft_vocab;

  uint32_t d0_target =
      eagle3_argmax_target_from_draft_logits(eagle3, last_pos_logits, draft_vocab, target_vocab);

  const int N_DRAFT = 5;
  std::vector<int64_t> draft_target_ids;
  draft_target_ids.push_back(d0_target);

  dynamic_tensor prenorm_cur = eagle3.get_prenorm();

  for (int j = 1; j < N_DRAFT; ++j) {
    int64_t last_idx = prenorm_cur.dim(1) - 1;
    size_t offset_bytes =
        static_cast<size_t>(last_idx) * static_cast<size_t>(eagle3_H) * sizeof(float);
    dynamic_tensor g_embd_next = prenorm_cur.make_view({1, 1, eagle3_H}, offset_bytes);

    eagle3.decode({static_cast<uint32_t>(draft_target_ids.back())}, g_embd_next,
                  static_cast<int64_t>(n_prompt + j - 1));

    const auto& step_logits = eagle3.get_logits();
    uint32_t dj_target = eagle3_argmax_target_from_draft_logits(
        eagle3, step_logits.data_ptr<float>(), draft_vocab, target_vocab);

    draft_target_ids.push_back(dj_target);
    prenorm_cur = eagle3.get_prenorm();
  }
  eagle3.stop();

  std::cout << "  Step | Draft | Target | Result\n";
  std::cout << "  -----|-------|--------|-------\n";

  int n_accepted = 0;
  bool still_accepting = true;

  for (int j = 0; j < N_DRAFT; ++j) {
    if (j == 0) {
      target.next(id_last);
    } else {
      target.next(static_cast<uint32_t>(draft_target_ids[j - 1]));
    }
    const auto& tgt_step_logits = target.get_logits();
    uint32_t t_j = argmax_f32(tgt_step_logits.data(), static_cast<int64_t>(tgt_step_logits.size()));

    bool accepted = still_accepting && (static_cast<uint32_t>(draft_target_ids[j]) == t_j);
    if (!accepted) still_accepting = false;
    if (accepted) ++n_accepted;

    std::cout << "  " << j << "    | " << draft_target_ids[j] << " | " << t_j << " | "
              << (accepted ? "ACCEPT" : "REJECT") << "\n";
  }

  target.stop();

  const double accept_rate = static_cast<double>(n_accepted) / static_cast<double>(N_DRAFT);
  std::cout << "\nAcceptance: " << n_accepted << "/" << N_DRAFT << " = "
            << static_cast<int>(accept_rate * 100.0 + 0.5) << "%\n";
  return 0;
}
