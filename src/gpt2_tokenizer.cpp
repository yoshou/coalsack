#include "gpt2_tokenizer.h"

#include <algorithm>
#include <climits>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace coalsack {

struct pair_hash {
  size_t operator()(const std::pair<std::string, std::string>& p) const {
    return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
  }
};

struct gpt2_tokenizer::impl {
  std::vector<std::string> vocab;
  std::unordered_map<std::string, uint32_t> token_to_id;
  std::vector<std::pair<std::string, std::string>> bpe_merges;
  std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;

  // Special/control tokens sorted by length (longest first) for greedy matching
  std::vector<std::pair<std::string, uint32_t>> special_tokens;

  uint32_t bos_id = 0;
  uint32_t eos_id = 0;

  bool add_bos = false;
  bool add_eos = false;

  bool loaded = false;

  std::string bytes_to_unicode_char(uint8_t byte) const {
    static const std::vector<int> byte_encoder = []() {
      std::vector<int> b2u(256);
      int n = 0;
      for (int b = 0; b < 256; ++b) {
        if ((b >= '!' && b <= '~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
          b2u[b] = b;
        } else {
          b2u[b] = 256 + n;
          n++;
        }
      }
      return b2u;
    }();

    int unicode_val = byte_encoder[byte];
    std::string result;
    if (unicode_val < 128) {
      result += static_cast<char>(unicode_val);
    } else {
      result += static_cast<char>(0xC0 | (unicode_val >> 6));
      result += static_cast<char>(0x80 | (unicode_val & 0x3F));
    }
    return result;
  }

  uint8_t unicode_char_to_byte(const std::string& uc) const {
    static const std::unordered_map<int, uint8_t> unicode_decoder = []() {
      std::unordered_map<int, uint8_t> u2b;
      int n = 0;
      for (int b = 0; b < 256; ++b) {
        if ((b >= '!' && b <= '~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
          u2b[b] = b;
        } else {
          u2b[256 + n] = b;
          n++;
        }
      }
      return u2b;
    }();

    int unicode_val;
    if (uc.size() == 1) {
      unicode_val = static_cast<uint8_t>(uc[0]);
    } else if (uc.size() == 2) {
      unicode_val =
          ((static_cast<uint8_t>(uc[0]) & 0x1F) << 6) | (static_cast<uint8_t>(uc[1]) & 0x3F);
    } else {
      return 0;
    }

    auto it = unicode_decoder.find(unicode_val);
    return (it != unicode_decoder.end()) ? it->second : 0;
  }

  std::vector<std::string> byte_encode_text(const std::string& text) const {
    std::vector<std::string> result;
    for (uint8_t byte : text) {
      result.push_back(bytes_to_unicode_char(byte));
    }
    return result;
  }

  std::string byte_decode_tokens(const std::vector<std::string>& tokens) const {
    std::string result;
    for (const auto& token : tokens) {
      for (size_t i = 0; i < token.size();) {
        if (i + 1 < token.size() && (static_cast<uint8_t>(token[i]) & 0xC0) == 0xC0) {
          std::string uc = token.substr(i, 2);
          result += unicode_char_to_byte(uc);
          i += 2;
        } else {
          result += unicode_char_to_byte(token.substr(i, 1));
          i += 1;
        }
      }
    }
    return result;
  }

  std::pair<std::string, std::string> get_best_merge(const std::vector<std::string>& word) const {
    std::pair<std::string, std::string> best_pair;
    int min_rank = INT_MAX;

    for (size_t i = 0; i + 1 < word.size(); ++i) {
      auto pair = std::make_pair(word[i], word[i + 1]);
      auto it = bpe_ranks.find(pair);
      if (it != bpe_ranks.end() && it->second < min_rank) {
        min_rank = it->second;
        best_pair = pair;
      }
    }

    return (min_rank == INT_MAX) ? std::make_pair("", "") : best_pair;
  }

  std::vector<std::string> apply_bpe(std::vector<std::string> word) const {
    if (word.size() <= 1) {
      return word;
    }

    while (true) {
      auto pair = get_best_merge(word);
      if (pair.first.empty()) {
        break;
      }

      std::vector<std::string> new_word;
      size_t i = 0;
      while (i < word.size()) {
        if (i + 1 < word.size() && word[i] == pair.first && word[i + 1] == pair.second) {
          new_word.push_back(pair.first + pair.second);
          i += 2;
        } else {
          new_word.push_back(word[i]);
          i += 1;
        }
      }
      word = std::move(new_word);
    }

    return word;
  }
};

gpt2_tokenizer::gpt2_tokenizer() : pimpl_(std::make_unique<impl>()) {}

gpt2_tokenizer::~gpt2_tokenizer() = default;

bool gpt2_tokenizer::load_from_gguf(const gguf_loader& loader) {
  if (!loader.is_loaded()) {
    std::cerr << "Error: GGUF loader not loaded\n";
    return false;
  }

  pimpl_->vocab = loader.get_array_string("tokenizer.ggml.tokens");
  if (pimpl_->vocab.empty()) {
    std::cerr << "Error: No tokens found in GGUF\n";
    return false;
  }

  for (size_t i = 0; i < pimpl_->vocab.size(); ++i) {
    pimpl_->token_to_id[pimpl_->vocab[i]] = i;
  }

  auto merges_str = loader.get_array_string("tokenizer.ggml.merges");
  for (const auto& merge : merges_str) {
    size_t space_pos = merge.find(' ');
    if (space_pos != std::string::npos) {
      std::string first = merge.substr(0, space_pos);
      std::string second = merge.substr(space_pos + 1);
      pimpl_->bpe_merges.push_back({first, second});
    }
  }

  for (size_t i = 0; i < pimpl_->bpe_merges.size(); ++i) {
    pimpl_->bpe_ranks[pimpl_->bpe_merges[i]] = i;
  }

  auto bos = loader.get_uint32("tokenizer.ggml.bos_token_id");
  auto eos = loader.get_uint32("tokenizer.ggml.eos_token_id");
  pimpl_->bos_id = bos.value_or(0);
  pimpl_->eos_id = eos.value_or(0);

  pimpl_->add_bos = loader.get_bool("tokenizer.ggml.add_bos_token").value_or(false);
  pimpl_->add_eos = loader.get_bool("tokenizer.ggml.add_eos_token").value_or(false);

  // Load token types and register special/control tokens
  auto token_types = loader.get_array_int32("tokenizer.ggml.token_type");
  for (size_t i = 0; i < token_types.size() && i < pimpl_->vocab.size(); ++i) {
    // token_type 3 = control token
    if (token_types[i] == 3) {
      pimpl_->special_tokens.push_back({pimpl_->vocab[i], static_cast<uint32_t>(i)});
    }
  }
  // Sort by length descending for greedy matching
  std::sort(pimpl_->special_tokens.begin(), pimpl_->special_tokens.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  pimpl_->loaded = true;
  std::cout << "Tokenizer loaded: " << pimpl_->vocab.size() << " tokens, "
            << pimpl_->bpe_merges.size() << " merges, " << pimpl_->special_tokens.size()
            << " special tokens\n";

  if (pimpl_->add_bos) {
    std::cout << "  tokenizer.ggml.add_bos_token: true (bos_id=" << pimpl_->bos_id << ")\n";
  }
  if (pimpl_->add_eos) {
    std::cout << "  tokenizer.ggml.add_eos_token: true (eos_id=" << pimpl_->eos_id << ")\n";
  }

  return true;
}

std::vector<uint32_t> gpt2_tokenizer::encode(const std::string& text) const {
  if (!pimpl_->loaded) {
    return {};
  }

  std::vector<uint32_t> ids;

  // Split text by special tokens, apply BPE only to non-special segments
  size_t pos = 0;
  while (pos < text.size()) {
    // Try to match a special token at current position
    bool found_special = false;
    for (const auto& [token_str, token_id] : pimpl_->special_tokens) {
      if (pos + token_str.size() <= text.size() &&
          text.compare(pos, token_str.size(), token_str) == 0) {
        ids.push_back(token_id);
        pos += token_str.size();
        found_special = true;
        break;
      }
    }
    if (found_special) {
      continue;
    }

    // Find the next special token to determine segment boundary
    size_t next_special = text.size();
    for (const auto& [token_str, token_id] : pimpl_->special_tokens) {
      size_t found = text.find(token_str, pos);
      if (found != std::string::npos && found < next_special) {
        next_special = found;
      }
    }

    // Encode the non-special segment with BPE
    std::string segment = text.substr(pos, next_special - pos);
    auto byte_tokens = pimpl_->byte_encode_text(segment);
    auto bpe_tokens = pimpl_->apply_bpe(byte_tokens);

    for (const auto& token : bpe_tokens) {
      auto it = pimpl_->token_to_id.find(token);
      if (it != pimpl_->token_to_id.end()) {
        ids.push_back(it->second);
      }
    }

    pos = next_special;
  }

  return ids;
}

std::string gpt2_tokenizer::decode(const std::vector<uint32_t>& tokens) const {
  if (!pimpl_->loaded) {
    return "";
  }

  std::vector<std::string> token_strings;
  for (uint32_t id : tokens) {
    if (id < pimpl_->vocab.size()) {
      token_strings.push_back(pimpl_->vocab[id]);
    }
  }

  return pimpl_->byte_decode_tokens(token_strings);
}

uint32_t gpt2_tokenizer::bos_token_id() const { return pimpl_->bos_id; }

uint32_t gpt2_tokenizer::eos_token_id() const { return pimpl_->eos_id; }

bool gpt2_tokenizer::add_bos_token() const { return pimpl_->add_bos; }

bool gpt2_tokenizer::add_eos_token() const { return pimpl_->add_eos; }

size_t gpt2_tokenizer::vocab_size() const { return pimpl_->vocab.size(); }

}  // namespace coalsack
