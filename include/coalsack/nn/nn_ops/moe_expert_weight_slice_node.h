#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "coalsack/nn/nn_op_node.h"

namespace coalsack {

// MoE Expert Weight Slice Node
// Extracts a single expert's weights from 3D weight tensors into 2D/1D slices
// Creates zero-copy views using dynamic_tensor's view/offset support
//
// Input expectations (7 inputs, all experts):
// - w_up_3d: [num_experts, expert_ffn_dim, hidden_dim] FLOAT16
// - w_gate_3d: [num_experts, expert_ffn_dim, hidden_dim] FLOAT16
// - w_down_3d: [num_experts, hidden_dim, expert_ffn_dim] FLOAT16
// - b_up_2d: [num_experts, expert_ffn_dim] FLOAT32
// - b_gate_2d: [num_experts, expert_ffn_dim] FLOAT32
// - b_down_2d: [num_experts, hidden_dim] FLOAT32
// - router_output: [batch, seq_len, top_k, 2] FLOAT32
//
// Output (7 fields for single expert):
// - w_up_2d: [expert_ffn_dim, hidden_dim] FLOAT16
// - w_gate_2d: [expert_ffn_dim, hidden_dim] FLOAT16
// - w_down_2d: [hidden_dim, expert_ffn_dim] FLOAT16
// - b_up_1d: [expert_ffn_dim] FLOAT32
// - b_gate_1d: [expert_ffn_dim] FLOAT32
// - b_down_1d: [hidden_dim] FLOAT32
// - router_output: passthrough
class moe_expert_weight_slice_node : public graph_node {
 private:
  graph_edge_ptr output_;
  int expert_id_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string node_name_;

 public:
  explicit moe_expert_weight_slice_node(int expert_id)
      : graph_node(), output_(std::make_shared<graph_edge>(this)), expert_id_(expert_id) {
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "moe_expert_weight_slice"; }

  void set_input_names(const std::vector<std::string>& names) { input_names_ = names; }
  void set_output_names(const std::vector<std::string>& names) { output_names_ = names; }
  void set_node_name(const std::string& name) { node_name_ = name; }

  int get_expert_id() const { return expert_id_; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping moe_expert_weight_slice [expert_{}] (Frame: {})", expert_id_,
                    frame_number);

      // Create error output with all 7 fields
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (const auto& name : output_names_) {
        fields[name] = nullptr;
      }

      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing moe_expert_weight_slice [expert_{}] (Frame: {})", expert_id_,
                  frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      // Extract input tensors from result_message
      dynamic_tensor w_up_3d = get_tensor_from_result_message(result_msg, input_names_[0]);
      dynamic_tensor w_gate_3d = get_tensor_from_result_message(result_msg, input_names_[1]);
      dynamic_tensor w_down_3d = get_tensor_from_result_message(result_msg, input_names_[2]);
      dynamic_tensor b_up_2d = get_tensor_from_result_message(result_msg, input_names_[3]);
      dynamic_tensor b_gate_2d = get_tensor_from_result_message(result_msg, input_names_[4]);
      dynamic_tensor b_down_2d = get_tensor_from_result_message(result_msg, input_names_[5]);
      dynamic_tensor router_output = get_tensor_from_result_message(result_msg, input_names_[6]);

      // Validate shapes
      if (w_up_3d.ndim() != 3 || w_gate_3d.ndim() != 3 || w_down_3d.ndim() != 3) {
        throw std::runtime_error("Weight tensors must be 3D [num_experts, ...]");
      }
      if (b_up_2d.ndim() != 2 || b_gate_2d.ndim() != 2 || b_down_2d.ndim() != 2) {
        throw std::runtime_error("Bias tensors must be 2D [num_experts, ...]");
      }

      int64_t num_experts = w_up_3d.dim(0);
      if (expert_id_ < 0 || expert_id_ >= num_experts) {
        throw std::runtime_error("expert_id " + std::to_string(expert_id_) + " out of range [0, " +
                                 std::to_string(num_experts) + ")");
      }

      // Extract dimensions
      int64_t expert_ffn_dim = w_up_3d.dim(1);
      int64_t hidden_dim = w_up_3d.dim(2);

      // Calculate byte offsets (assuming C-order: expert_id is the outermost dimension)
      size_t dtype_size_fp16 = 2;  // FLOAT16
      size_t dtype_size_fp32 = 4;  // FLOAT32

      // For w_up/w_gate: [num_experts, expert_ffn_dim, hidden_dim]
      // Offset for expert_id: expert_id * (expert_ffn_dim * hidden_dim) * sizeof(fp16)
      size_t w_up_offset = expert_id_ * expert_ffn_dim * hidden_dim * dtype_size_fp16;
      size_t w_gate_offset = w_up_offset;

      // For w_down: [num_experts, hidden_dim, expert_ffn_dim]
      size_t w_down_offset = expert_id_ * hidden_dim * expert_ffn_dim * dtype_size_fp16;

      // For biases: [num_experts, dim]
      size_t b_up_offset = expert_id_ * expert_ffn_dim * dtype_size_fp32;
      size_t b_gate_offset = b_up_offset;
      size_t b_down_offset = expert_id_ * hidden_dim * dtype_size_fp32;

      // Create 2D/1D views
      dynamic_tensor w_up_2d = w_up_3d.make_view({expert_ffn_dim, hidden_dim}, w_up_offset);
      dynamic_tensor w_gate_2d = w_gate_3d.make_view({expert_ffn_dim, hidden_dim}, w_gate_offset);
      dynamic_tensor w_down_2d = w_down_3d.make_view({hidden_dim, expert_ffn_dim}, w_down_offset);
      dynamic_tensor b_up_1d = b_up_2d.make_view({expert_ffn_dim}, b_up_offset);
      dynamic_tensor b_gate_1d = b_gate_2d.make_view({expert_ffn_dim}, b_gate_offset);
      dynamic_tensor b_down_1d = b_down_2d.make_view({hidden_dim}, b_down_offset);

      // Create output message with 7 fields
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_names_[0]] = std::make_shared<dynamic_tensor_message>(w_up_2d);
      fields[output_names_[1]] = std::make_shared<dynamic_tensor_message>(w_gate_2d);
      fields[output_names_[2]] = std::make_shared<dynamic_tensor_message>(w_down_2d);
      fields[output_names_[3]] = std::make_shared<dynamic_tensor_message>(b_up_1d);
      fields[output_names_[4]] = std::make_shared<dynamic_tensor_message>(b_gate_1d);
      fields[output_names_[5]] = std::make_shared<dynamic_tensor_message>(b_down_1d);
      fields[output_names_[6]] = std::make_shared<dynamic_tensor_message>(router_output);

      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);

    } catch (const std::exception& e) {
      spdlog::error("Error in moe_expert_weight_slice [expert_{}]: {}", expert_id_, e.what());

      std::unordered_map<std::string, graph_message_ptr> fields;
      for (const auto& name : output_names_) {
        fields[name] = nullptr;
      }

      output_msg = result_message::error(fields, std::string(e.what()));
      output_msg->set_frame_number(frame_number);
    }

    output_->send(output_msg);
  }
};

}  // namespace coalsack
