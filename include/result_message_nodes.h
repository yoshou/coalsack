#pragma once

#include <spdlog/spdlog.h>

#include <map>
#include <memory>
#include <unordered_map>

#include "graph_proc.h"
#include "result_message.h"
#include "syncer.h"

namespace coalsack {

class result_message_sync_node : public graph_node {
 private:
  using syncer_type = stream_syncer<graph_message_ptr, std::string, frame_number>;
  using stream_id_type = std::string;

  syncer_type syncer_;
  graph_edge_ptr output_;
  std::vector<stream_id_type> initial_ids_;

 public:
  result_message_sync_node()
      : graph_node(), syncer_(), output_(std::make_shared<graph_edge>(this)), initial_ids_() {
    set_output(output_);
  }

  std::string get_proc_name() const override { return "result_message_sync_node"; }

  void set_initial_ids(const std::vector<std::string>& ids) { initial_ids_ = ids; }

  const std::vector<stream_id_type>& get_initial_ids() const { return initial_ids_; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(initial_ids_);
  }

  void initialize() override {
    syncer_.start(
        std::make_shared<typename syncer_type::callback_type>(
            [this](const std::map<std::string, graph_message_ptr>& frames) {
              std::unordered_map<std::string, graph_message_ptr> merged_fields;

              uint64_t frame_number = 0;
              double timestamp = 0.0;
              bool has_error = false;

              for (const auto& [input_name, message] : frames) {
                if (auto result_msg = std::dynamic_pointer_cast<result_message>(message)) {
                  if (!result_msg->is_ok()) {
                    has_error = true;
                  }
                  const auto& fields = result_msg->get_fields();

                  auto it = fields.find(input_name);
                  if (it == fields.end()) {
                    throw std::runtime_error("sync_node: field '" + input_name + "' not found");
                  }
                  merged_fields[input_name] = it->second;
                  frame_number = result_msg->get_frame_number();
                  timestamp = result_msg->get_timestamp();
                }
              }

              std::shared_ptr<result_message> synced;
              if (has_error) {
                for (const auto& id : initial_ids_) {
                  if (merged_fields.find(id) == merged_fields.end()) {
                    merged_fields[id] = nullptr;
                  }
                }
                synced = result_message::error(merged_fields, "Input error");
              } else {
                synced = result_message::ok(merged_fields);
              }
              synced->set_frame_number(frame_number);
              synced->set_timestamp(timestamp);

              output_->send(synced);
            }),
        initial_ids_);
  }

  void run() override {}

  void process(std::string input_name, graph_message_ptr message) override {
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (!result_msg) {
      spdlog::warn("result_message_sync_node received non-result_message for {}", input_name);
      return;
    }

    syncer_.sync(input_name, result_msg, frame_number(result_msg->get_frame_number()));
  }
};

class result_field_extractor_node : public graph_node {
 private:
  std::unordered_map<std::string, graph_edge_ptr> field_outputs_;

 public:
  result_field_extractor_node() : graph_node() {}

  std::string get_proc_name() const override { return "result_field_extractor_node"; }

  graph_edge_ptr add_output(const std::string& field_name) {
    auto it = field_outputs_.find(field_name);
    if (it != field_outputs_.end()) {
      return it->second;
    }

    auto edge = std::make_shared<graph_edge>(this);
    edge->set_name(field_name);
    field_outputs_[field_name] = edge;
    set_output(edge, field_name);
    return edge;
  }

  void process(std::string input_name, graph_message_ptr message) override {
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (!result_msg) {
      return;
    }

    if (!result_msg->is_ok()) {
      for (const auto& [field_name, edge] : field_outputs_) {
        std::unordered_map<std::string, graph_message_ptr> fields;
        fields[field_name] = nullptr;
        auto error_msg = result_message::error(fields, "Input error");
        error_msg->set_frame_number(result_msg->get_frame_number());
        error_msg->set_timestamp(result_msg->get_timestamp());
        edge->send(error_msg);
      }
      return;
    }

    for (const auto& [field_name, edge] : field_outputs_) {
      auto field = result_msg->get_field(field_name);
      if (!field) {
        throw std::runtime_error("Field '" + field_name + "' not found");
      }

      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[field_name] = field;
      auto output = result_message::ok(fields);
      output->set_frame_number(result_msg->get_frame_number());
      output->set_timestamp(result_msg->get_timestamp());

      edge->send(output);
    }
  }
};

}  // namespace coalsack
