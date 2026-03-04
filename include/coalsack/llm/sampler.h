#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace coalsack {

// Token sampler for speculative decoding.
//
// Sampling pipeline:
//   temperature == 0  →  greedy argmax
//   temperature > 0   →  temp-scale → top-k → softmax-renorm → discrete_distribution
//
// The internal RNG is seeded once at construction and advances with each call to sample().
// Call reset() to replay from the initial seed.
class sampler {
 public:
  struct config {
    float temperature = 0.0f;  // 0 = greedy
    int top_k = 0;             // 0 = no top-k (all vocab)
    float top_p = 1.0f;        // 1 = no top-p (reserved)
    uint64_t seed = 0;
  };

  explicit sampler() : cfg_(config{}), rng_(0) {}
  explicit sampler(const config& cfg) : cfg_(cfg), rng_(cfg.seed) {}

  // Reset RNG to the initial seed (allows deterministic replay)
  void reset() { rng_.seed(cfg_.seed); }

  // Sample a token from logits[0..vocab_size).
  // temperature == 0 → argmax (greedy, no RNG consumed)
  // temperature > 0  → top-k stochastic
  uint32_t sample(const float* logits, int64_t vocab_size) {
    if (cfg_.temperature <= 0.0f) {
      return argmax(logits, vocab_size);
    }

    // Build (scaled_logit, original_index) pairs
    std::vector<std::pair<float, int64_t>> cands(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
      cands[i] = {logits[i] / cfg_.temperature, i};
    }

    // Partial sort: keep top-k at front
    const int k = (cfg_.top_k > 0 && cfg_.top_k < static_cast<int>(vocab_size))
                      ? cfg_.top_k
                      : static_cast<int>(vocab_size);
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    cands.resize(static_cast<size_t>(k));

    // Softmax over top-k (numerically stable: subtract max first)
    const float max_l = cands[0].first;
    float sum = 0.0f;
    for (auto& [l, _] : cands) {
      l = std::exp(l - max_l);
      sum += l;
    }
    std::vector<float> probs(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
      probs[i] = cands[i].first / sum;
    }

    // Sample with persistent RNG
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    const int idx = dist(rng_);
    return static_cast<uint32_t>(cands[static_cast<size_t>(idx)].second);
  }

  // Returns the probability of the top-1 token after applying temperature scaling and top-k.
  // Used for p_min early-exit in draft generation.
  float get_top1_prob_after_topk(const float* logits, int64_t vocab_size) const {
    const int k = (cfg_.top_k > 0 && cfg_.top_k < static_cast<int>(vocab_size))
                      ? cfg_.top_k
                      : static_cast<int>(vocab_size);
    const float temp = (cfg_.temperature > 0.0f) ? cfg_.temperature : 1.0f;

    // Find top-k values after temp scaling
    std::vector<float> scaled(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
      scaled[i] = logits[i] / temp;
    }
    std::partial_sort(scaled.begin(), scaled.begin() + k, scaled.end(), std::greater<float>());
    scaled.resize(static_cast<size_t>(k));

    // Softmax
    const float max_l = scaled[0];
    float sum = 0.0f;
    for (auto& v : scaled) {
      v = std::exp(v - max_l);
      sum += v;
    }
    return scaled[0] / sum;  // max prob (after top-k renorm)
  }

  // Utility: compute full softmax of logits[0..n) and return probability vector.
  static std::vector<float> softmax(const float* logits, int64_t n) {
    const float max_l = *std::max_element(logits, logits + n);
    std::vector<float> probs(static_cast<size_t>(n));
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
      probs[i] = std::exp(logits[i] - max_l);
      sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
  }

  // Whether using greedy sampling (temperature == 0)
  bool is_greedy() const { return cfg_.temperature <= 0.0f; }

  // Result of speculative verification.
  struct verify_result {
    int n_accepted;             // number of accepted draft tokens (0..n_draft)
    uint32_t correction_token;  // token to emit at the first rejected position;
                                // when n_accepted==n_draft this is the bonus token
                                // sampled from the last position's target logits
  };

  // Speculative decoding verification (Leviathan et al. 2023).
  //
  //  draft_tokens[i]    : i-th draft token in TARGET vocab space
  //  target_logits_all  : [n_draft * target_vocab] — position i is the target logits
  //                       produced when draft_tokens[i-1] (or id_last for i==0) was fed
  //  n_draft            : number of draft tokens to verify
  //  target_vocab       : target model vocabulary size
  //  target_sampler     : selects correction token; temperature==0 → greedy argmax
  //
  // Greedy target (temperature==0): accept iff argmax(target[i]) == draft_tokens[i]
  // Stochastic target:              accept with probability min(1, p_target/p_draft)
  static verify_result speculative_verify(const uint32_t* draft_tokens,
                                          const float* target_logits_all, int n_draft,
                                          int64_t target_vocab, sampler& target_sampler) {
    for (int i = 0; i < n_draft; ++i) {
      const float* logits_i =
          target_logits_all + static_cast<size_t>(i) * static_cast<size_t>(target_vocab);
      const uint32_t t_i = target_sampler.sample(logits_i, target_vocab);
      if (t_i != draft_tokens[i]) {
        return {i, t_i};
      }
    }
    // All accepted: bonus token from logits AFTER the last draft token (index n_draft)
    // Caller must provide n_draft+1 logit vectors.
    const float* last_l =
        target_logits_all + static_cast<size_t>(n_draft) * static_cast<size_t>(target_vocab);
    return {n_draft, target_sampler.sample(last_l, target_vocab)};
  }

  // Public static argmax for fallback use.
  static uint32_t argmax_static(const float* data, int64_t n) { return argmax(data, n); }

 private:
  static uint32_t argmax(const float* data, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i) {
      if (data[i] > data[best]) best = i;
    }
    return static_cast<uint32_t>(best);
  }

  config cfg_;
  std::mt19937 rng_;
};

}  // namespace coalsack
