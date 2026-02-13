#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "dynamic_mx_tensor_message.h"
#include "dynamic_tensor.h"
#include "dynamic_tensor_message.h"
#include "graph_proc.h"
#include "moe_weight_provider.h"
#include "nn_op_node.h"
#include "result_message.h"

#include <spdlog/spdlog.h>

namespace coalsack {

class moe_weight_fetch_node : public graph_node {
 private:
  std::shared_ptr<moe_weight_provider> weight_provider_;
  std::string layer_prefix_;
  std::unordered_map<int, graph_edge_ptr> expert_outputs_;

  // Extract selected expert IDs from router_output tensor
  // Router output shape: [batch, seq_len, top_k, 2]
  // Last dimension: [expert_index, weight]
  std::set<int> extract_selected_experts(const dynamic_tensor& router_tensor) const {
    std::set<int> selected_experts;
    
    if (router_tensor.ndim() != 4) {
      return selected_experts;
    }

    const float* router_data = router_tensor.data_ptr<float>();
    int64_t batch = router_tensor.dim(0);
    int64_t seq_len = router_tensor.dim(1);
    int64_t top_k = router_tensor.dim(2);
    int64_t last_dim = router_tensor.dim(3);

    if (last_dim != 2) {
      return selected_experts;
    }

    // Parse router output to extract unique expert IDs
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t k = 0; k < top_k; ++k) {
          int64_t idx = ((b * seq_len + s) * top_k + k) * 2;
          int expert_id = static_cast<int>(router_data[idx]);
          selected_experts.insert(expert_id);
        }
      }
    }

    return selected_experts;
  }

 public:
  moe_weight_fetch_node(std::shared_ptr<moe_weight_provider> provider,
                        const std::string& layer_prefix)
      : graph_node(), weight_provider_(std::move(provider)), layer_prefix_(layer_prefix) {
    if (!weight_provider_) {
      throw std::invalid_argument("weight_provider cannot be null");
    }
  }

  std::string get_proc_name() const override { return "moe_weight_fetch_node"; }

  graph_edge_ptr add_expert_output(int expert_id) {
    auto it = expert_outputs_.find(expert_id);
    if (it != expert_outputs_.end()) {
      return it->second;
    }

    auto edge = std::make_shared<graph_edge>(this);
    edge->set_name("expert_" + std::to_string(expert_id));
    expert_outputs_[expert_id] = edge;
    set_output(edge, "expert_" + std::to_string(expert_id));
    return edge;
  }

  void process(std::string input_name, graph_message_ptr message) override {
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (!result_msg) {
      return;
    }

    uint64_t frame_number = result_msg->get_frame_number();
    double timestamp = result_msg->get_timestamp();

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping moe_weight_fetch_node [{}] (Frame: {})", layer_prefix_, frame_number);
      
      for (const auto& [expert_id, edge] : expert_outputs_) {
        std::unordered_map<std::string, graph_message_ptr> fields;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_up"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_gate"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_down"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_up"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_gate"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_down"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".router_out"] = nullptr;
        
        auto error_msg = result_message::error(fields, "Input error");
        error_msg->set_frame_number(frame_number);
        error_msg->set_timestamp(timestamp);
        edge->send(error_msg);
      }
      return;
    }

    spdlog::trace("Executing moe_weight_fetch_node [{}] (Frame: {})", layer_prefix_, frame_number);

    // Parse router output
    dynamic_tensor router_tensor;
    std::shared_ptr<dynamic_tensor_message> router_tensor_msg;
    std::set<int> selected_experts;
    
    try {
      router_tensor = get_tensor_from_result_message(result_msg, layer_prefix_ + ".router_out");
      router_tensor_msg = std::make_shared<dynamic_tensor_message>(router_tensor);
      selected_experts = extract_selected_experts(router_tensor);

    } catch (const std::exception& e) {
      spdlog::error("moe_weight_fetch_node [{}]: {}", layer_prefix_, e.what());
      for (const auto& [expert_id, edge] : expert_outputs_) {
        std::unordered_map<std::string, graph_message_ptr> fields;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_up"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_gate"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_down"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_up"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_gate"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_down"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".router_out"] = nullptr;
        
        auto error_msg = result_message::error(fields, e.what());
        error_msg->set_frame_number(frame_number);
        error_msg->set_timestamp(timestamp);
        edge->send(error_msg);
      }
      return;
    }

    // Process each expert
    for (const auto& [expert_id, edge] : expert_outputs_) {
      bool is_selected = selected_experts.find(expert_id) != selected_experts.end();

      if (!is_selected) {
        dynamic_tensor empty_tensor(dtype::float32, {0});
        auto empty_tensor_msg = std::make_shared<dynamic_tensor_message>(empty_tensor);
        
        std::unordered_map<std::string, graph_message_ptr> fields;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_up"] = empty_tensor_msg;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_gate"] = empty_tensor_msg;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_down"] = empty_tensor_msg;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_up"] = empty_tensor_msg;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_gate"] = empty_tensor_msg;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_down"] = empty_tensor_msg;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".router_out"] = router_tensor_msg;

        auto empty_msg = result_message::ok(fields);
        empty_msg->set_frame_number(frame_number);
        empty_msg->set_timestamp(timestamp);
        edge->send(empty_msg);
        continue;
      }

      std::shared_ptr<result_message> output_msg;
      
      try {
        auto w_up = weight_provider_->get(
            layer_prefix_ + ".ffn_up_exps.weight", expert_id);
        auto w_gate = weight_provider_->get(
            layer_prefix_ + ".ffn_gate_exps.weight", expert_id);
        auto w_down = weight_provider_->get(
            layer_prefix_ + ".ffn_down_exps.weight", expert_id);
        
        auto b_up = weight_provider_->get(
            layer_prefix_ + ".ffn_up_exps.bias", expert_id);
        auto b_gate = weight_provider_->get(
            layer_prefix_ + ".ffn_gate_exps.bias", expert_id);
        auto b_down = weight_provider_->get(
            layer_prefix_ + ".ffn_down_exps.bias", expert_id);

        std::unordered_map<std::string, graph_message_ptr> fields;
        
        // Convert variant to message
        auto make_message = [](const auto& variant) -> graph_message_ptr {
          if (std::holds_alternative<dynamic_tensor>(variant)) {
            auto msg = std::make_shared<dynamic_tensor_message>();
            msg->set_tensor(std::get<dynamic_tensor>(variant));
            return msg;
          } else {
            auto msg = std::make_shared<dynamic_mx_tensor_message>();
            msg->set_mx_tensor(std::get<dynamic_mx_tensor>(variant));
            return msg;
          }
        };
        
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_up"] = make_message(w_up);
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_gate"] = make_message(w_gate);
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_down"] = make_message(w_down);
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_up"] = make_message(b_up);
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_gate"] = make_message(b_gate);
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_down"] = make_message(b_down);
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".router_out"] = 
            router_tensor_msg;

        output_msg = result_message::ok(fields);
        output_msg->set_frame_number(frame_number);
        output_msg->set_timestamp(timestamp);

      } catch (const std::exception& e) {
        spdlog::error("moe_weight_fetch_node [{}] expert {}: {}", layer_prefix_, expert_id, e.what());
        std::unordered_map<std::string, graph_message_ptr> fields;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_up"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_gate"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".w_down"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_up"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_gate"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".b_down"] = nullptr;
        fields[layer_prefix_ + ".expert_" + std::to_string(expert_id) + ".router_out"] = nullptr;
        
        output_msg = result_message::error(fields, e.what());
        output_msg->set_frame_number(frame_number);
        output_msg->set_timestamp(timestamp);
      }
      
      edge->send(output_msg);
    }
  }
};

}  // namespace coalsack
