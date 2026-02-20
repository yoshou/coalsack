#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coalsack/nn/model_io_nodes.h"
#include "coalsack/nn/nn_nodes.h"
#include "coalsack/nn/nn_ops/dense_ffn_node.h"
#include "coalsack/nn/nn_ops/grouped_attention_node.h"
#include "coalsack/nn/nn_ops/matmul_transpose_mixed_node.h"
#include "coalsack/nn/nn_ops/moe_weight_fetch_node.h"
#include "coalsack/nn/nn_ops/reshape_node.h"
#include "coalsack/nn/result_message_nodes.h"
#include "coalsack/tensor/dynamic_tensor.h"

namespace coalsack {

// Helper class that encapsulates all sync-node-pair creation patterns used
// when building the transformer computation graph. Each method creates a
// result_message_sync_node + actual-node pair, registers both with the
// target graph and returns the output graph_edge_ptr of the actual node.
class graph_builder {
 public:
  // Configuration values forwarded from llm_engine::impl.
  struct config {
    int64_t max_seq_len = 8192;
    float rope_freq_base = 10000.0f;
    float rope_scaling_factor = 1.0f;
    std::string rope_scaling_type = "none";
    int64_t rope_scaling_orig_ctx = 0;
    bool use_norm_rope = false;
    int64_t num_experts = 0;
    int64_t expert_top_k = 0;
    bool use_sigmoid_gating = false;
    bool weight_before_ffn = false;
  };

  graph_builder(std::shared_ptr<subgraph> graph, const config& cfg)
      : graph_(std::move(graph)), cfg_(cfg) {}

  // sync + embedding_lookup_node
  graph_edge_ptr make_embedding(graph_edge_ptr input_ids, const std::string& input_ids_name,
                                graph_edge_ptr weight, const std::string& weight_name,
                                const std::string& out_name, const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(input_ids, input_ids_name);
    sync->set_input(weight, weight_name);
    sync->set_initial_ids({input_ids_name, weight_name});
    graph_->add_node(sync);
    auto node = std::make_shared<embedding_lookup_node>();
    node->set_input(sync->get_output(), "default");
    node->set_input_names(input_ids_name, weight_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // layer_scheduler_node — single input, no sync required
  graph_edge_ptr make_layer_scheduler(graph_edge_ptr in) {
    auto node = std::make_shared<layer_scheduler_node>();
    node->set_input(in, "default");
    graph_->add_node(node);
    return node->get_output();
  }

  // moe_weight_fetch_node + per-expert output edges
  std::vector<graph_edge_ptr> make_moe_weight_fetch(std::shared_ptr<moe_weight_provider> provider,
                                                    const std::string& layer_prefix,
                                                    graph_edge_ptr router_out, bool has_bias) {
    auto node = std::make_shared<moe_weight_fetch_node>(provider, layer_prefix);
    node->set_has_bias(has_bias);
    node->set_input(router_out, layer_prefix + ".router_out");
    graph_->add_node(node);
    std::vector<graph_edge_ptr> outputs;
    for (int i = 0; i < cfg_.num_experts; ++i) {
      outputs.push_back(node->add_expert_output(i));
    }
    return outputs;
  }

  // sync + matmul_transpose_mixed_node
  graph_edge_ptr make_matmul(graph_edge_ptr in, const std::string& in_name, graph_edge_ptr weight,
                             const std::string& weight_name, const std::string& out_name,
                             const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(weight, weight_name);
    sync->set_initial_ids({in_name, weight_name});
    graph_->add_node(sync);
    auto node = std::make_shared<matmul_transpose_mixed_node>();
    node->set_input(sync->get_output(), "default");
    node->set_input_names(in_name, weight_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync + add_node
  graph_edge_ptr make_add(graph_edge_ptr in1, const std::string& name1, graph_edge_ptr in2,
                          const std::string& name2, const std::string& out_name,
                          const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in1, name1);
    sync->set_input(in2, name2);
    sync->set_initial_ids({name1, name2});
    graph_->add_node(sync);
    auto node = std::make_shared<add_node>();
    node->set_input(sync->get_output(), "default");
    node->set_input_names(name1, name2);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync + mul_node
  graph_edge_ptr make_mul(graph_edge_ptr in1, const std::string& name1, graph_edge_ptr in2,
                          const std::string& name2, const std::string& out_name,
                          const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in1, name1);
    sync->set_input(in2, name2);
    sync->set_initial_ids({name1, name2});
    graph_->add_node(sync);
    auto node = std::make_shared<mul_node>();
    node->set_input(sync->get_output(), "default");
    node->set_input_names(name1, name2);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync + rmsnorm_node (epsilon = 1e-5)
  graph_edge_ptr make_rmsnorm(graph_edge_ptr in, const std::string& in_name, graph_edge_ptr weight,
                              const std::string& weight_name, const std::string& out_name,
                              const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(weight, weight_name);
    sync->set_initial_ids({in_name, weight_name});
    graph_->add_node(sync);
    auto node = std::make_shared<rmsnorm_node>();
    node->set_epsilon(1e-5f);
    node->set_input(sync->get_output(), "default");
    node->set_input_names(in_name, weight_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // l2norm_node — single input, no sync required
  graph_edge_ptr make_l2norm(graph_edge_ptr in, const std::string& in_name,
                             const std::string& out_name, const std::string& node_name) {
    auto node = std::make_shared<l2norm_node>();
    node->set_input(in, "default");
    node->set_input_name(in_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync + reshape_node
  graph_edge_ptr make_reshape(graph_edge_ptr in, const std::string& in_name, graph_edge_ptr shape,
                              const std::string& shape_name, const std::string& out_name,
                              const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(shape, shape_name);
    sync->set_initial_ids({in_name, shape_name});
    graph_->add_node(sync);
    auto node = std::make_shared<reshape_node>();
    node->set_input(sync->get_output(), "default");
    node->set_input_names(in_name, shape_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // transpose_node — single input, no sync required
  graph_edge_ptr make_transpose(graph_edge_ptr in, const std::string& in_name,
                                const std::vector<int64_t>& perm, const std::string& out_name,
                                const std::string& node_name) {
    auto node = std::make_shared<transpose_node>();
    node->set_perm(perm);
    node->set_input(in, "default");
    node->set_input_name(in_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // constant_node holding int64 shape data; frame_ref provides batch linkage.
  // No sync required.
  graph_edge_ptr make_shape_const(const dynamic_tensor& shape_data, const std::string& const_name,
                                  graph_edge_ptr frame_ref) {
    auto node = std::make_shared<constant_node>(shape_data, const_name);
    node->set_input(frame_ref, "default");
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync + rope_node (rope_freqs_edge may be nullptr)
  graph_edge_ptr make_rope(graph_edge_ptr in, const std::string& in_name, graph_edge_ptr pos_ids,
                           graph_edge_ptr rfreqs, int64_t hdim, const std::string& out_name,
                           const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(pos_ids, "position_ids");
    std::vector<std::string> input_names = {in_name, "position_ids"};
    if (rfreqs) {
      sync->set_input(rfreqs, "rope_freqs.weight");
      input_names.push_back("rope_freqs.weight");
    }
    sync->set_initial_ids(input_names);
    graph_->add_node(sync);
    auto node = std::make_shared<rope_node>();
    node->set_config(hdim, cfg_.max_seq_len, cfg_.rope_freq_base, cfg_.rope_scaling_factor,
                     cfg_.rope_scaling_type, cfg_.rope_scaling_orig_ctx);
    node->set_neox_style(!cfg_.use_norm_rope);
    node->set_input(sync->get_output(), "default");
    node->set_input_names(input_names);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync(q,k,v) + grouped_attention_node
  graph_edge_ptr make_grouped_attn(graph_edge_ptr q, const std::string& q_name, graph_edge_ptr k,
                                   const std::string& k_name, graph_edge_ptr v,
                                   const std::string& v_name, int64_t num_q_heads,
                                   int64_t num_kv_heads, int64_t head_dim, int64_t sliding_window,
                                   const std::optional<dynamic_tensor>& attn_sinks,
                                   const std::string& out_name, const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(q, q_name);
    sync->set_input(k, k_name);
    sync->set_input(v, v_name);
    sync->set_initial_ids({q_name, k_name, v_name});
    graph_->add_node(sync);
    auto attn_node = std::make_shared<grouped_attention_node>();
    attn_node->set_config(num_q_heads, num_kv_heads, head_dim, sliding_window, attn_sinks);
    attn_node->set_input(sync->get_output(), "default");
    attn_node->set_input_names({q_name, k_name, v_name});
    attn_node->set_output_name(out_name);
    attn_node->set_node_name(node_name);
    graph_->add_node(attn_node);
    return attn_node->get_output("default");
  }

  // sync(in,gate,up,down) + dense_ffn_node
  graph_edge_ptr make_dense_ffn(graph_edge_ptr in, const std::string& in_name, graph_edge_ptr gate,
                                const std::string& gate_name, graph_edge_ptr up,
                                const std::string& up_name, graph_edge_ptr down,
                                const std::string& down_name, const std::string& out_name,
                                const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(gate, gate_name);
    sync->set_input(up, up_name);
    sync->set_input(down, down_name);
    sync->set_initial_ids({in_name, gate_name, up_name, down_name});
    graph_->add_node(sync);
    auto node = std::make_shared<dense_ffn_node>();
    node->set_input(sync->get_output(), "default");
    node->set_input_names({in_name, gate_name, up_name, down_name});
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync(in, weight[, bias]) + moe_router_node
  graph_edge_ptr make_moe_router(graph_edge_ptr in, const std::string& in_name,
                                 graph_edge_ptr weight, const std::string& weight_name,
                                 graph_edge_ptr bias,  // nullptr if absent
                                 const std::string& bias_name, const std::string& out_name,
                                 const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(weight, weight_name);
    std::vector<std::string> input_names = {in_name, weight_name};
    if (bias) {
      sync->set_input(bias, bias_name);
      input_names.push_back(bias_name);
    }
    sync->set_initial_ids(input_names);
    graph_->add_node(sync);
    auto node = std::make_shared<moe_router_node>();
    node->set_config(cfg_.num_experts, cfg_.expert_top_k);
    node->set_sigmoid_gating(cfg_.use_sigmoid_gating);
    node->set_input(sync->get_output(), "default");
    node->set_input_names(input_names);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // variadic sync + expert_mlp_node
  graph_edge_ptr make_expert_mlp(int expert_id, graph_edge_ptr in, const std::string& in_name,
                                 graph_edge_ptr fetch_edge, const std::string& ep, bool has_bias,
                                 expert_mlp_node::activation_type act_type) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(fetch_edge, ep + ".w_up");
    sync->set_input(fetch_edge, ep + ".w_gate");
    sync->set_input(fetch_edge, ep + ".w_down");
    std::vector<std::string> input_names = {in_name, ep + ".w_up", ep + ".w_gate", ep + ".w_down"};
    if (has_bias) {
      sync->set_input(fetch_edge, ep + ".b_up");
      sync->set_input(fetch_edge, ep + ".b_gate");
      sync->set_input(fetch_edge, ep + ".b_down");
      input_names.push_back(ep + ".b_up");
      input_names.push_back(ep + ".b_gate");
      input_names.push_back(ep + ".b_down");
    }
    sync->set_input(fetch_edge, ep + ".router_out");
    input_names.push_back(ep + ".router_out");
    sync->set_initial_ids(input_names);
    graph_->add_node(sync);
    auto node = std::make_shared<expert_mlp_node>(expert_id);
    node->set_activation_type(act_type);
    node->set_weight_before_ffn(cfg_.weight_before_ffn);
    node->set_input(sync->get_output(), "default");
    node->set_input_names(input_names);
    node->set_output_name(ep + "_out");
    node->set_node_name(ep);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // variadic sync + expert_merge_node
  graph_edge_ptr make_expert_merge(graph_edge_ptr router_out, const std::string& router_out_name,
                                   const std::vector<graph_edge_ptr>& expert_outs,
                                   const std::string& lprefix, const std::string& out_name) {
    std::vector<std::string> input_names = {router_out_name};
    for (int i = 0; i < cfg_.num_experts; ++i) {
      input_names.push_back(lprefix + ".expert_" + std::to_string(i) + "_out");
    }
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(router_out, router_out_name);
    for (int i = 0; i < cfg_.num_experts; ++i) {
      sync->set_input(expert_outs[i], input_names[i + 1]);
    }
    sync->set_initial_ids(input_names);
    graph_->add_node(sync);
    auto node = std::make_shared<expert_merge_node>();
    node->set_config(cfg_.num_experts, cfg_.expert_top_k);
    node->set_weight_before_ffn(cfg_.weight_before_ffn);
    node->set_input(sync->get_output(), "default");
    node->set_input_names(input_names);
    node->set_output_name(out_name);
    node->set_node_name(lprefix + ".expert_merge");
    graph_->add_node(node);
    return node->get_output("default");
  }

  // If bias_key exists in lweights, appends make_add for the bias and returns
  // {biased_edge, biased_out_name}. Otherwise returns {in, in_name} unchanged.
  std::pair<graph_edge_ptr, std::string> apply_optional_bias(
      graph_edge_ptr in, const std::string& in_name,
      const std::unordered_map<std::string, graph_edge_ptr>& lweights, const std::string& bias_key,
      const std::string& biased_out_name, const std::string& node_name) {
    if (lweights.count(bias_key)) {
      return {make_add(in, in_name, lweights.at(bias_key), bias_key, biased_out_name, node_name),
              biased_out_name};
    }
    return {in, in_name};
  }

 private:
  std::shared_ptr<subgraph> graph_;
  config cfg_;
};

}  // namespace coalsack
