/// @file text_message.h
/// @brief Message carrying a UTF-8 text string.
/// @ingroup messages
#pragma once

#include <string>

#include "coalsack/core/graph_message.h"

namespace coalsack {

/// @defgroup messages Message Types
/// @brief Typed message classes that carry data between graph nodes.
/// @{

/// @brief Message carrying a UTF-8 encoded text string.
/// @details Wraps a single @c std::string payload.  Use @c set_text() / @c get_text()
///          to read and write the content.
///
/// @par Fields
/// - text (std::string) — the UTF-8 string payload
class text_message : public graph_message {
  std::string text;

 public:
  text_message() : text() {}

  void set_text(std::string text) { this->text = text; }
  std::string get_text() const { return text; }
  static std::string get_type() { return "text"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(text);
  }
};

/// @}
}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::text_message, coalsack::graph_message)
