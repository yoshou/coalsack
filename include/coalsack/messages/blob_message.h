/// @file blob_message.h
/// @brief Message carrying an opaque binary byte buffer.
/// @ingroup messages
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "coalsack/core/graph_message.h"

namespace coalsack {

/// @brief Message carrying a variable-length byte array (e.g. encoded image or audio).
///
/// @par Fields
/// - data (std::vector<uint8_t>) — the raw byte payload
class blob_message : public graph_message {
  std::vector<uint8_t> data;

 public:
  blob_message() : data() {}

  void set_data(const std::vector<uint8_t>& data) { this->data = data; }
  void set_data(std::vector<uint8_t>&& data) { this->data = std::move(data); }
  const std::vector<uint8_t>& get_data() const { return data; }
  static std::string get_type() { return "blob"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(data);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::blob_message, coalsack::graph_message)
