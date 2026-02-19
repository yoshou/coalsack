#include "coalsack/llm/chat_template.h"

#include <sstream>

namespace coalsack {

struct message {
  enum class role { system, user, assistant };
  role role_type;
  std::string content;
  std::string channel;
};

struct chat_template::impl {
  std::vector<message> messages;

  static constexpr const char* MARKER_START = "<|start|>";
  static constexpr const char* MARKER_END = "<|end|>";
  static constexpr const char* MARKER_MESSAGE = "<|message|>";
  static constexpr const char* MARKER_CHANNEL = "<|channel|>";

  std::string role_to_string(message::role r) const {
    switch (r) {
      case message::role::system:
        return "system";
      case message::role::user:
        return "user";
      case message::role::assistant:
        return "assistant";
      default:
        return "";
    }
  }

  std::string format_message(const message& msg) const {
    std::ostringstream oss;
    // Format: <|start|>{role}<|message|>{content}<|end|>
    oss << MARKER_START << role_to_string(msg.role_type) << MARKER_MESSAGE;
    oss << msg.content << MARKER_END;
    return oss.str();
  }
};

chat_template::chat_template() : pimpl_(std::make_unique<impl>()) {}

chat_template::~chat_template() = default;

void chat_template::add_system(const std::string& content) {
  message msg;
  msg.role_type = message::role::system;
  msg.content = content;
  pimpl_->messages.push_back(msg);
}

void chat_template::add_user(const std::string& content) {
  message msg;
  msg.role_type = message::role::user;
  msg.content = content;
  pimpl_->messages.push_back(msg);
}

void chat_template::add_assistant(const std::string& content) {
  message msg;
  msg.role_type = message::role::assistant;
  msg.content = content;
  msg.channel = "final";
  pimpl_->messages.push_back(msg);
}

std::string chat_template::build_prompt() const {
  std::ostringstream oss;

  for (const auto& msg : pimpl_->messages) {
    oss << pimpl_->format_message(msg);
  }

  // Format: <|start|>assistant<|channel|>final<|message|>
  // This matches Python version
  oss << pimpl_->MARKER_START << "assistant" << pimpl_->MARKER_CHANNEL << "final"
      << pimpl_->MARKER_MESSAGE;

  return oss.str();
}

std::string chat_template::extract_final_response(const std::string& generated_text) const {
  std::string final_marker = std::string(pimpl_->MARKER_CHANNEL) + "final" + pimpl_->MARKER_MESSAGE;

  size_t start_pos = generated_text.find(final_marker);
  if (start_pos == std::string::npos) {
    return "";
  }

  start_pos += final_marker.length();

  size_t end_pos = generated_text.find(pimpl_->MARKER_END, start_pos);
  if (end_pos == std::string::npos) {
    return generated_text.substr(start_pos);
  }

  return generated_text.substr(start_pos, end_pos - start_pos);
}

void chat_template::clear() { pimpl_->messages.clear(); }

size_t chat_template::message_count() const { return pimpl_->messages.size(); }

}  // namespace coalsack
