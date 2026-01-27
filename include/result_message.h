#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "graph_proc.h"
#include "graph_proc_img.h"

namespace coalsack {

class result_message : public frame_message_base {
 private:
  bool ok_;
  std::unordered_map<std::string, graph_message_ptr> fields_;
  graph_message_ptr error_;

 public:
  static std::shared_ptr<result_message> ok(const std::string& field_name,
                                            graph_message_ptr value) {
    auto result = std::make_shared<result_message>();
    result->ok_ = true;
    result->fields_[field_name] = value;
    if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(value)) {
      result->set_frame_number(frame_msg->get_frame_number());
      result->set_timestamp(frame_msg->get_timestamp());
    }
    return result;
  }

  static std::shared_ptr<result_message> ok(
      const std::unordered_map<std::string, graph_message_ptr>& fields) {
    auto result = std::make_shared<result_message>();
    result->ok_ = true;
    result->fields_ = fields;
    if (!fields.empty()) {
      if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(fields.begin()->second)) {
        result->set_frame_number(frame_msg->get_frame_number());
        result->set_timestamp(frame_msg->get_timestamp());
      }
    }
    return result;
  }

  void add_field(const std::string& name, graph_message_ptr message) { fields_[name] = message; }

  graph_message_ptr get_field(const std::string& name) const {
    auto it = fields_.find(name);
    if (it != fields_.end()) {
      return it->second;
    }
    return nullptr;
  }

  const std::unordered_map<std::string, graph_message_ptr>& get_fields() const { return fields_; }
  static std::shared_ptr<result_message> error(
      const std::unordered_map<std::string, graph_message_ptr>& fields,
      const std::string& message) {
    auto result = std::make_shared<result_message>();
    result->ok_ = false;
    result->fields_ = fields;

    auto err_msg = std::make_shared<object_message>();
    auto text_msg = std::make_shared<text_message>();
    text_msg->set_text(message);
    err_msg->add_field("message", text_msg);
    result->error_ = err_msg;

    return result;
  }

  bool is_ok() const { return ok_; }
  bool is_error() const { return !ok_; }

  graph_message_ptr get_value() const {
    if (!ok_) {
      throw std::logic_error("Attempted to get value from error result");
    }
    if (!fields_.empty()) {
      return fields_.begin()->second;
    }
    return nullptr;
  }

  graph_message_ptr get_error() const {
    if (ok_) {
      throw std::logic_error("Attempted to get error from ok result");
    }
    return error_;
  }

  std::string get_error_message() const {
    if (ok_) return "";

    auto err_obj = std::dynamic_pointer_cast<object_message>(error_);
    if (!err_obj) return "Unknown error";

    auto msg_field = err_obj->get_field("message");
    auto text_msg = std::dynamic_pointer_cast<text_message>(msg_field);
    if (text_msg) {
      return text_msg->get_text();
    }

    return "Unknown error";
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    frame_message_base::serialize(archive);
    archive(ok_);
    if (ok_) {
      archive(fields_);
    } else {
      archive(error_);
    }
  }

  result_message() : frame_message_base(), ok_(true), fields_(), error_(nullptr) {}
};

}  // namespace coalsack

COALSACK_REGISTER_MESSAGE(coalsack::result_message, coalsack::frame_message_base)
