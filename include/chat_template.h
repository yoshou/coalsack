#pragma once

#include <memory>
#include <string>
#include <vector>

namespace coalsack {

class chat_template {
 public:
  chat_template();
  ~chat_template();

  void add_system(const std::string& content);
  void add_user(const std::string& content);
  void add_assistant(const std::string& content);

  std::string build_prompt() const;

  std::string extract_final_response(const std::string& generated_text) const;

  void clear();

  size_t message_count() const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};

}  // namespace coalsack
