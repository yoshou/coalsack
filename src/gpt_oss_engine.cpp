#include "gpt_oss_engine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <random>

#include "gguf_dequant.h"
#include "gguf_multi_loader.h"
#include "gpt2_tokenizer.h"
#include "graph_proc.h"
#include "model_io_nodes.h"
#include "nn_nodes.h"
#include "nn_ops/add_node.h"
#include "nn_ops/constant_node.h"
#include "nn_ops/grouped_attention_node.h"
#include "nn_ops/matmul_transpose_mixed_node.h"
#include "nn_ops/reshape_node.h"
#include "result_message_nodes.h"

namespace coalsack {

struct gpt_oss_engine::impl {
  // Components
  std::unique_ptr<gguf_multi_loader> loader;
  std::unique_ptr<gpt2_tokenizer> tokenizer;
  std::shared_ptr<subgraph> graph;
  std::unique_ptr<graph_proc> proc;

  // I/O nodes
  std::shared_ptr<model_input_node> input_node;
  std::shared_ptr<model_output_node> output_node;

  // Constant nodes (for weight tensors, need frame source connection)
  std::vector<std::shared_ptr<constant_node>> constant_nodes;
  
  // Attention nodes (for KV cache management)
  std::vector<std::shared_ptr<grouped_attention_node>> attention_nodes;

  // Position tracking for RoPE
  int64_t cached_position_ = 0;

  // Model config (from GGUF metadata)
  int64_t num_layers = 24;
  int64_t hidden_dim = 2880;
  int64_t num_q_heads = 64;
  int64_t num_kv_heads = 8;
  int64_t head_dim = 45;  // hidden_dim / num_q_heads
  int64_t num_experts = 32;
  int64_t expert_top_k = 4;
  int64_t expert_ffn_dim = 2880;
  int64_t vocab_size = 201088;
  int64_t max_seq_len = 8192;
  int64_t kv_cache_size = std::numeric_limits<int64_t>::max();

  // Attention config
  int64_t attention_sliding_window = 0;

  // RoPE config
  float rope_freq_base = 150000.0f;
  float rope_scaling_factor = 32.0f;
  std::string rope_scaling_type = "none";
  int64_t rope_scaling_orig_ctx = 0;

  // Weights (loaded from GGUF)
  std::unordered_map<std::string, dynamic_tensor> weights;

  bool loaded = false;
};

gpt_oss_engine::gpt_oss_engine() : pimpl_(std::make_unique<impl>()) {
  pimpl_->loader = std::make_unique<gguf_multi_loader>();
  pimpl_->tokenizer = std::make_unique<gpt2_tokenizer>();
  pimpl_->kv_cache_size = std::numeric_limits<int64_t>::max();
}

gpt_oss_engine::gpt_oss_engine(const config& cfg) : pimpl_(std::make_unique<impl>()) {
  pimpl_->loader = std::make_unique<gguf_multi_loader>();
  pimpl_->tokenizer = std::make_unique<gpt2_tokenizer>();
  pimpl_->kv_cache_size = cfg.kv_cache_size;
}

gpt_oss_engine::~gpt_oss_engine() = default;

bool gpt_oss_engine::load(const std::string& gguf_path) {
  return load(std::vector<std::string>{gguf_path});
}

bool gpt_oss_engine::load(const std::vector<std::string>& gguf_paths) {
  if (pimpl_->loaded) {
    throw std::runtime_error("Model already loaded");
  }

  if (gguf_paths.empty()) {
    throw std::runtime_error("No GGUF files specified");
  }

  if (!pimpl_->loader->load(gguf_paths)) {
    throw std::runtime_error("Failed to load GGUF file(s)");
  }

  if (!pimpl_->tokenizer->load_from_gguf(*pimpl_->loader)) {
    throw std::runtime_error("Failed to load tokenizer");
  }

  load_config_from_gguf();

  load_weights_from_gguf();

  build_transformer_graph();

  pimpl_->loaded = true;

  return true;
}

void gpt_oss_engine::load_config_from_gguf() {
  auto& loader = *pimpl_->loader;

  // Helper for required parameters
  auto get_required_uint32 = [&](const std::string& suffix) -> uint32_t {
    std::string key = "gpt-oss." + suffix;
    if (auto v = loader.get_uint32(key)) return *v;
    throw std::runtime_error("Missing required parameter: " + key);
  };

  auto get_required_float = [&](const std::string& suffix) -> float {
    std::string key = "gpt-oss." + suffix;
    if (auto v = loader.get_float32(key)) return *v;
    throw std::runtime_error("Missing required parameter: " + key);
  };

  auto get_optional_uint32 = [&](const std::string& suffix) -> std::optional<uint32_t> {
    return loader.get_uint32("gpt-oss." + suffix);
  };

  auto get_optional_string = [&](const std::string& suffix) -> std::optional<std::string> {
    return loader.get_string("gpt-oss." + suffix);
  };

  pimpl_->num_layers = get_required_uint32("block_count");
  pimpl_->hidden_dim = get_required_uint32("embedding_length");
  pimpl_->num_q_heads = get_required_uint32("attention.head_count");
  pimpl_->num_kv_heads = get_required_uint32("attention.head_count_kv");
  
  if (auto val = get_optional_uint32("vocab_size")) {
    pimpl_->vocab_size = *val;
  } else if (pimpl_->tokenizer) {
    pimpl_->vocab_size = pimpl_->tokenizer->vocab_size();
  } else {
    throw std::runtime_error("vocab_size missing and tokenizer not loaded");
  }

  pimpl_->max_seq_len = get_required_uint32("context_length");

  if (auto v = get_optional_uint32("attention.sliding_window")) {
    pimpl_->attention_sliding_window = static_cast<int64_t>(*v);
  }

  if (auto val = get_optional_uint32("expert_count")) {
    pimpl_->num_experts = *val;
  }
  if (auto val = get_optional_uint32("expert_used_count")) {
    pimpl_->expert_top_k = *val;
  }
  if (auto val = get_optional_uint32("expert_feed_forward_length")) {
    pimpl_->expert_ffn_dim = *val;
  }

  std::optional<uint32_t> head_dim_opt;
  if (auto val = get_optional_uint32("attention.head_dim")) {
    head_dim_opt = val;
  } else if (auto val = get_optional_uint32("attention.key_length")) {
    head_dim_opt = val;
  }
  
  if (head_dim_opt) {
    pimpl_->head_dim = static_cast<int64_t>(*head_dim_opt);
  } else {
    pimpl_->head_dim = pimpl_->hidden_dim / pimpl_->num_q_heads;
  }

  // RoPE config
  pimpl_->rope_freq_base = get_required_float("rope.freq_base");
  pimpl_->rope_scaling_factor = get_required_float("rope.scaling.factor");
  if (auto v = get_optional_string("rope.scaling.type")) {
    pimpl_->rope_scaling_type = *v;
  }
  if (auto v = get_optional_uint32("rope.scaling.original_context_length")) {
    pimpl_->rope_scaling_orig_ctx = static_cast<int64_t>(*v);
  }
}

void gpt_oss_engine::load_weights_from_gguf() {
  auto& loader = *pimpl_->loader;
  const auto& tensor_names = loader.get_tensor_names();
  const auto& shard_paths = loader.get_shard_paths();

  // Open all shard files
  std::vector<std::ifstream> files;
  for (const auto& path : shard_paths) {
    files.emplace_back(path, std::ios::binary);
    if (!files.back()) {
      throw std::runtime_error("Failed to open file for tensor loading: " + path);
    }
  }

  size_t loaded_count = 0;
  size_t dequantized_count = 0;

  for (const auto& name : tensor_names) {
    auto info_opt = loader.get_tensor_info(name);
    if (!info_opt) {
      throw std::runtime_error("Failed to get tensor info for: " + name);
    }
    const auto& info = *info_opt;

    // Check shard index validity
    if (info.shard_idx >= files.size()) {
      throw std::runtime_error("Invalid shard index for tensor: " + name);
    }

    auto& file = files[info.shard_idx];

    std::vector<int64_t> shape;
    for (int i = info.shape.size() - 1; i >= 0; --i) {
      shape.push_back(static_cast<int64_t>(info.shape[i]));
    }

    int64_t numel = 1;
    for (auto dim : shape) {
      numel *= dim;
    }

    file.seekg(info.offset);
    if (!file) {
      throw std::runtime_error("Failed to seek to tensor data for: " + name);
    }

    std::vector<uint8_t> raw_data(info.size);
    file.read(reinterpret_cast<char*>(raw_data.data()), info.size);
    if (!file) {
      throw std::runtime_error("Failed to read tensor data for: " + name);
    }

    dtype output_dtype = (info.type == ggml_type::F16 || info.type == ggml_type::MXFP4)
                             ? dtype::float16
                             : dtype::float32;

    dynamic_tensor tensor(output_dtype, shape);

    if (info.type == ggml_type::F32) {
      std::memcpy(tensor.data_ptr<float>(), raw_data.data(), info.size);
      ++loaded_count;
    } else if (info.type == ggml_type::F16) {
      std::memcpy(tensor.data_ptr<uint16_t>(), raw_data.data(), info.size);
      ++loaded_count;
    } else if (info.type == ggml_type::MXFP4) {
      if (!dequantize_mxfp4_to_fp16(raw_data.data(), tensor.data_ptr<uint16_t>(), numel)) {
        throw std::runtime_error("Failed to convert MXFP4 to float16 for: " + name);
      }
      ++dequantized_count;
    } else {
      if (!dequantize_tensor(raw_data.data(), tensor.data_ptr<float>(), numel, info.type)) {
        throw std::runtime_error("Unsupported quantization type for: " + name);
      }
      ++dequantized_count;
    }

    pimpl_->weights[name] = std::move(tensor);
  }

  std::vector<std::string> required_weights = {"token_embd.weight", "output.weight"};
  for (const auto& req : required_weights) {
    if (pimpl_->weights.find(req) == pimpl_->weights.end()) {
      throw std::runtime_error("Required weight not found: " + req);
    }
  }

  const auto& tok_w = pimpl_->weights.at("token_embd.weight");
  const auto& out_w = pimpl_->weights.at("output.weight");

  if (tok_w.ndim() != 2 || out_w.ndim() != 2) {
    throw std::runtime_error("Embedding weights must be 2D");
  }

  if (tok_w.dim(1) != pimpl_->hidden_dim || out_w.dim(1) != pimpl_->hidden_dim) {
    throw std::runtime_error("Embedding weight dim(1) should be hidden_dim=" +
                             std::to_string(pimpl_->hidden_dim));
  }

  if (pimpl_->vocab_size > 0) {
    if (tok_w.dim(0) != pimpl_->vocab_size || out_w.dim(0) != pimpl_->vocab_size) {
      throw std::runtime_error("Embedding weight dim(0) should be vocab_size=" +
                               std::to_string(pimpl_->vocab_size));
    }
  }
}

void gpt_oss_engine::build_transformer_graph() {
  pimpl_->graph = std::make_shared<subgraph>();

  // Create I/O nodes
  pimpl_->input_node = std::make_shared<model_input_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  std::vector<std::string> required_weights = {
      "token_embd.weight",
      "output_norm.weight",
      "output.weight",
  };

  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_prefix = "blk." + std::to_string(layer);
    required_weights.push_back(layer_prefix + ".attn_norm.weight");
    required_weights.push_back(layer_prefix + ".attn_q.weight");
    required_weights.push_back(layer_prefix + ".attn_q.bias");
    required_weights.push_back(layer_prefix + ".attn_k.weight");
    required_weights.push_back(layer_prefix + ".attn_k.bias");
    required_weights.push_back(layer_prefix + ".attn_v.weight");
    required_weights.push_back(layer_prefix + ".attn_v.bias");
    required_weights.push_back(layer_prefix + ".attn_output.weight");
    required_weights.push_back(layer_prefix + ".attn_output.bias");
    required_weights.push_back(layer_prefix + ".post_attention_norm.weight");
    required_weights.push_back(layer_prefix + ".ffn_gate_inp.weight");
    required_weights.push_back(layer_prefix + ".ffn_gate_inp.bias");
    required_weights.push_back(layer_prefix + ".ffn_up_exps.weight");
    required_weights.push_back(layer_prefix + ".ffn_up_exps.bias");
    required_weights.push_back(layer_prefix + ".ffn_gate_exps.weight");
    required_weights.push_back(layer_prefix + ".ffn_gate_exps.bias");
    required_weights.push_back(layer_prefix + ".ffn_down_exps.weight");
    required_weights.push_back(layer_prefix + ".ffn_down_exps.bias");
  }

  for (const auto& name : required_weights) {
    auto it = pimpl_->weights.find(name);
    if (it == pimpl_->weights.end()) {
      throw std::runtime_error("Required weight not found: " + name);
    }
    pimpl_->input_node->set_tensor(name, it->second);
  }

  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  auto input_ids_edge = extractor->add_output("input_ids");
  auto position_ids_edge = extractor->add_output("position_ids");  // Add position_ids
  auto token_embd_weight_edge_raw = extractor->add_output("token_embd.weight");
  auto output_norm_weight_edge = extractor->add_output("output_norm.weight");
  auto output_weight_edge = extractor->add_output("output.weight");
  auto output_weight_edge_transposed = output_weight_edge;
  auto token_embd_weight_edge = token_embd_weight_edge_raw;

  // Extract layer weights
  std::vector<std::unordered_map<std::string, graph_edge_ptr>> layer_weights(pimpl_->num_layers);
  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_prefix = "blk." + std::to_string(layer);
    std::vector<std::string> layer_weight_names = {
        layer_prefix + ".attn_norm.weight",
        layer_prefix + ".attn_q.weight",
        layer_prefix + ".attn_q.bias",
        layer_prefix + ".attn_k.weight",
        layer_prefix + ".attn_k.bias",
        layer_prefix + ".attn_v.weight",
        layer_prefix + ".attn_v.bias",
        layer_prefix + ".attn_output.weight",
        layer_prefix + ".attn_output.bias",
        layer_prefix + ".post_attention_norm.weight",
        layer_prefix + ".ffn_gate_inp.weight",
        layer_prefix + ".ffn_gate_inp.bias",
        layer_prefix + ".ffn_up_exps.weight",
        layer_prefix + ".ffn_up_exps.bias",
        layer_prefix + ".ffn_gate_exps.weight",
        layer_prefix + ".ffn_gate_exps.bias",
        layer_prefix + ".ffn_down_exps.weight",
        layer_prefix + ".ffn_down_exps.bias",
    };

    for (const auto& name : layer_weight_names) {
      if (pimpl_->weights.find(name) != pimpl_->weights.end()) {
        layer_weights[layer][name] = extractor->add_output(name);
      }
    }
  }

  // Embedding layer
  auto embedding = std::make_shared<embedding_lookup_node>();
  auto emb_sync = std::make_shared<result_message_sync_node>();
  emb_sync->set_input(input_ids_edge, "input_ids");
  emb_sync->set_input(token_embd_weight_edge, "token_embd.weight");
  emb_sync->set_initial_ids({"input_ids", "token_embd.weight"});
  pimpl_->graph->add_node(emb_sync);

  embedding->set_input(emb_sync->get_output(), "default");
  embedding->set_input_names("input_ids", "token_embd.weight");
  embedding->set_output_name("embeddings");
  embedding->set_node_name("embedding_lookup");
  pimpl_->graph->add_node(embedding);

  graph_edge_ptr current = embedding->get_output("default");
  std::string current_name = "embeddings";

  // Transformer layers
  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_str = std::to_string(layer);
    std::string layer_prefix = "blk." + layer_str;
    int64_t layer_head_dim = pimpl_->head_dim;
    graph_edge_ptr layer_input = current;
    std::string layer_input_name = current_name;

    // Input RMSNorm
    auto input_norm = std::make_shared<rmsnorm_node>();
    input_norm->set_epsilon(1e-5f);

    auto norm_sync = std::make_shared<result_message_sync_node>();
    norm_sync->set_input(current, current_name);
    norm_sync->set_input(layer_weights[layer][layer_prefix + ".attn_norm.weight"], layer_prefix + ".attn_norm.weight");
    norm_sync->set_initial_ids({current_name, layer_prefix + ".attn_norm.weight"});
    pimpl_->graph->add_node(norm_sync);

    input_norm->set_input(norm_sync->get_output(), "default");
    input_norm->set_input_names(current_name, layer_prefix + ".attn_norm.weight");
    input_norm->set_output_name(layer_prefix + ".attn_norm_out");
    input_norm->set_node_name(layer_prefix + ".attn_norm");
    pimpl_->graph->add_node(input_norm);

    graph_edge_ptr norm_out = input_norm->get_output("default");

    // Q projection
    auto q_proj = std::make_shared<matmul_transpose_mixed_node>();
    auto q_sync = std::make_shared<result_message_sync_node>();
    q_sync->set_input(norm_out, layer_prefix + ".attn_norm_out");
    q_sync->set_input(layer_weights[layer][layer_prefix + ".attn_q.weight"], layer_prefix + ".attn_q.weight");
    q_sync->set_initial_ids({layer_prefix + ".attn_norm_out", layer_prefix + ".attn_q.weight"});
    pimpl_->graph->add_node(q_sync);

    q_proj->set_input(q_sync->get_output(), "default");
    q_proj->set_input_names(layer_prefix + ".attn_norm_out", layer_prefix + ".attn_q.weight");
    q_proj->set_output_name(layer_prefix + ".q_proj_out");
    q_proj->set_node_name(layer_prefix + ".q_proj");
    pimpl_->graph->add_node(q_proj);
    
    // Q bias
    auto q_bias_add = std::make_shared<add_node>();
    auto q_bias_sync = std::make_shared<result_message_sync_node>();
    q_bias_sync->set_input(q_proj->get_output("default"), layer_prefix + ".q_proj_out");
    q_bias_sync->set_input(layer_weights[layer][layer_prefix + ".attn_q.bias"], layer_prefix + ".attn_q.bias");
    q_bias_sync->set_initial_ids({layer_prefix + ".q_proj_out", layer_prefix + ".attn_q.bias"});
    pimpl_->graph->add_node(q_bias_sync);
    
    q_bias_add->set_input(q_bias_sync->get_output(), "default");
    q_bias_add->set_input_names(layer_prefix + ".q_proj_out", layer_prefix + ".attn_q.bias");
    q_bias_add->set_output_name(layer_prefix + ".q_with_bias");
    q_bias_add->set_node_name(layer_prefix + ".q_bias");
    pimpl_->graph->add_node(q_bias_add);
    graph_edge_ptr q_out = q_bias_add->get_output("default");

    // K projection
    auto k_proj = std::make_shared<matmul_transpose_mixed_node>();
    auto k_sync = std::make_shared<result_message_sync_node>();
    k_sync->set_input(norm_out, layer_prefix + ".attn_norm_out");
    k_sync->set_input(layer_weights[layer][layer_prefix + ".attn_k.weight"], layer_prefix + ".attn_k.weight");
    k_sync->set_initial_ids({layer_prefix + ".attn_norm_out", layer_prefix + ".attn_k.weight"});
    pimpl_->graph->add_node(k_sync);

    k_proj->set_input(k_sync->get_output(), "default");
    k_proj->set_input_names(layer_prefix + ".attn_norm_out", layer_prefix + ".attn_k.weight");
    k_proj->set_output_name(layer_prefix + ".k_proj_out");
    k_proj->set_node_name(layer_prefix + ".k_proj");
    pimpl_->graph->add_node(k_proj);
    
    // K bias
    auto k_bias_add = std::make_shared<add_node>();
    auto k_bias_sync = std::make_shared<result_message_sync_node>();
    k_bias_sync->set_input(k_proj->get_output("default"), layer_prefix + ".k_proj_out");
    k_bias_sync->set_input(layer_weights[layer][layer_prefix + ".attn_k.bias"], layer_prefix + ".attn_k.bias");
    k_bias_sync->set_initial_ids({layer_prefix + ".k_proj_out", layer_prefix + ".attn_k.bias"});
    pimpl_->graph->add_node(k_bias_sync);
    
    k_bias_add->set_input(k_bias_sync->get_output(), "default");
    k_bias_add->set_input_names(layer_prefix + ".k_proj_out", layer_prefix + ".attn_k.bias");
    k_bias_add->set_output_name(layer_prefix + ".k_with_bias");
    k_bias_add->set_node_name(layer_prefix + ".k_bias");
    pimpl_->graph->add_node(k_bias_add);
    graph_edge_ptr k_out = k_bias_add->get_output("default");

    // V projection
    auto v_proj = std::make_shared<matmul_transpose_mixed_node>();
    auto v_sync = std::make_shared<result_message_sync_node>();
    v_sync->set_input(norm_out, layer_prefix + ".attn_norm_out");
    v_sync->set_input(layer_weights[layer][layer_prefix + ".attn_v.weight"], layer_prefix + ".attn_v.weight");
    v_sync->set_initial_ids({layer_prefix + ".attn_norm_out", layer_prefix + ".attn_v.weight"});
    pimpl_->graph->add_node(v_sync);

    v_proj->set_input(v_sync->get_output(), "default");
    v_proj->set_input_names(layer_prefix + ".attn_norm_out", layer_prefix + ".attn_v.weight");
    v_proj->set_output_name(layer_prefix + ".v_proj_out");
    v_proj->set_node_name(layer_prefix + ".v_proj");
    pimpl_->graph->add_node(v_proj);
    
    // V bias
    auto v_bias_add = std::make_shared<add_node>();
    auto v_bias_sync = std::make_shared<result_message_sync_node>();
    v_bias_sync->set_input(v_proj->get_output("default"), layer_prefix + ".v_proj_out");
    v_bias_sync->set_input(layer_weights[layer][layer_prefix + ".attn_v.bias"], layer_prefix + ".attn_v.bias");
    v_bias_sync->set_initial_ids({layer_prefix + ".v_proj_out", layer_prefix + ".attn_v.bias"});
    pimpl_->graph->add_node(v_bias_sync);
    
    v_bias_add->set_input(v_bias_sync->get_output(), "default");
    v_bias_add->set_input_names(layer_prefix + ".v_proj_out", layer_prefix + ".attn_v.bias");
    v_bias_add->set_output_name(layer_prefix + ".v_with_bias");
    v_bias_add->set_node_name(layer_prefix + ".v_bias");
    pimpl_->graph->add_node(v_bias_add);
    graph_edge_ptr v_out = v_bias_add->get_output("default");

    // 2.3 Reshape Q/K for RoPE (3D → 4D → transpose → RoPE → transpose → 4D → 3D)

    // Create shape constants for reshape operations
    // Q: [batch, seq, 4096] → [batch, seq, 64, 64] (auto-inferred from -1)
    dynamic_tensor q_reshape_shape(dtype::int64, {4});
    auto q_reshape_shape_data = q_reshape_shape.data_ptr<int64_t>();
    q_reshape_shape_data[0] = 0;                      // Copy batch from input[0]
    q_reshape_shape_data[1] = 0;                      // Copy seq_len from input[1]
    q_reshape_shape_data[2] = pimpl_->num_q_heads;   // 64
    q_reshape_shape_data[3] = -1;                     // Infer from total elements

    auto q_reshape_shape_const = std::make_shared<constant_node>(q_reshape_shape, layer_prefix + ".q_reshape_shape");
    q_reshape_shape_const->set_input(q_out, "default");
    pimpl_->graph->add_node(q_reshape_shape_const);
    auto q_reshape_shape_edge = q_reshape_shape_const->get_output("default");

    dynamic_tensor k_reshape_shape(dtype::int64, {4});
    auto k_reshape_shape_data = k_reshape_shape.data_ptr<int64_t>();
    k_reshape_shape_data[0] = 0;                      // Copy batch from input[0]
    k_reshape_shape_data[1] = 0;                      // Copy seq_len from input[1]
    k_reshape_shape_data[2] = pimpl_->num_kv_heads;  // 8
    k_reshape_shape_data[3] = -1;                     // Infer from total elements

    auto k_reshape_shape_const = std::make_shared<constant_node>(k_reshape_shape, layer_prefix + ".k_reshape_shape");
    k_reshape_shape_const->set_input(k_out, "default");
    pimpl_->graph->add_node(k_reshape_shape_const);
    auto k_reshape_shape_edge = k_reshape_shape_const->get_output("default");

    dynamic_tensor reshape_back_shape(dtype::int64, {3});
    auto reshape_back_shape_data = reshape_back_shape.data_ptr<int64_t>();
    reshape_back_shape_data[0] = 0;   // Copy batch
    reshape_back_shape_data[1] = 0;   // Copy seq_len
    reshape_back_shape_data[2] = -1;  // Infer from total elements

    auto reshape_back_shape_const = std::make_shared<constant_node>(reshape_back_shape, layer_prefix + ".reshape_back_shape");
    reshape_back_shape_const->set_input(q_out, "default");
    pimpl_->graph->add_node(reshape_back_shape_const);
    auto reshape_back_shape_edge = reshape_back_shape_const->get_output("default");

    // Process Q
    auto q_reshape_4d = std::make_shared<reshape_node>();
    auto q_reshape_4d_sync = std::make_shared<result_message_sync_node>();
    q_reshape_4d_sync->set_input(q_out, layer_prefix + ".q_with_bias");
    q_reshape_4d_sync->set_input(q_reshape_shape_edge, layer_prefix + ".q_reshape_shape");
    q_reshape_4d_sync->set_initial_ids({layer_prefix + ".q_with_bias", layer_prefix + ".q_reshape_shape"});
    pimpl_->graph->add_node(q_reshape_4d_sync);

    q_reshape_4d->set_input(q_reshape_4d_sync->get_output(), "default");
    q_reshape_4d->set_input_names(layer_prefix + ".q_with_bias", layer_prefix + ".q_reshape_shape");
    q_reshape_4d->set_output_name(layer_prefix + ".q_4d");
    q_reshape_4d->set_node_name(layer_prefix + ".q_reshape_4d");
    pimpl_->graph->add_node(q_reshape_4d);
    graph_edge_ptr q_4d = q_reshape_4d->get_output("default");

    auto q_transpose = std::make_shared<transpose_node>();
    q_transpose->set_perm({0, 2, 1, 3});
    q_transpose->set_input(q_4d, "default");
    q_transpose->set_input_name(layer_prefix + ".q_4d");
    q_transpose->set_output_name(layer_prefix + ".q_transposed");
    q_transpose->set_node_name(layer_prefix + ".q_transpose");
    pimpl_->graph->add_node(q_transpose);
    graph_edge_ptr q_transposed = q_transpose->get_output("default");

    auto rope_q = std::make_shared<rope_node>();
    rope_q->set_config(layer_head_dim, pimpl_->max_seq_len, pimpl_->rope_freq_base,
               pimpl_->rope_scaling_factor, pimpl_->rope_scaling_type,
               pimpl_->rope_scaling_orig_ctx);
    
    auto rope_q_sync = std::make_shared<result_message_sync_node>();
    rope_q_sync->set_input(q_transposed, layer_prefix + ".q_transposed");
    rope_q_sync->set_input(position_ids_edge, "position_ids");
    rope_q_sync->set_initial_ids({layer_prefix + ".q_transposed", "position_ids"});
    pimpl_->graph->add_node(rope_q_sync);
    
    rope_q->set_input(rope_q_sync->get_output(), "default");
    rope_q->set_input_names({layer_prefix + ".q_transposed", "position_ids"});
    rope_q->set_output_name(layer_prefix + ".q_rope");
    rope_q->set_node_name(layer_prefix + ".rope_q");
    pimpl_->graph->add_node(rope_q);
    graph_edge_ptr q_rope = rope_q->get_output("default");

    auto q_transpose_back = std::make_shared<transpose_node>();
    q_transpose_back->set_perm({0, 2, 1, 3});
    q_transpose_back->set_input(q_rope, "default");
    q_transpose_back->set_input_name(layer_prefix + ".q_rope");
    q_transpose_back->set_output_name(layer_prefix + ".q_rope_4d");
    q_transpose_back->set_node_name(layer_prefix + ".q_transpose_back");
    pimpl_->graph->add_node(q_transpose_back);
    graph_edge_ptr q_rope_4d = q_transpose_back->get_output("default");

    auto q_reshape_3d = std::make_shared<reshape_node>();
    auto q_reshape_3d_sync = std::make_shared<result_message_sync_node>();
    q_reshape_3d_sync->set_input(q_rope_4d, layer_prefix + ".q_rope_4d");
    q_reshape_3d_sync->set_input(reshape_back_shape_edge, layer_prefix + ".reshape_back_shape");
    q_reshape_3d_sync->set_initial_ids({layer_prefix + ".q_rope_4d", layer_prefix + ".reshape_back_shape"});
    pimpl_->graph->add_node(q_reshape_3d_sync);

    q_reshape_3d->set_input(q_reshape_3d_sync->get_output(), "default");
    q_reshape_3d->set_input_names(layer_prefix + ".q_rope_4d", layer_prefix + ".reshape_back_shape");
    q_reshape_3d->set_output_name(layer_prefix + ".q_rope_out");
    q_reshape_3d->set_node_name(layer_prefix + ".q_reshape_3d");
    pimpl_->graph->add_node(q_reshape_3d);
    graph_edge_ptr q_rope_out = q_reshape_3d->get_output("default");

    // Process K
    auto k_reshape_4d = std::make_shared<reshape_node>();
    auto k_reshape_4d_sync = std::make_shared<result_message_sync_node>();
    k_reshape_4d_sync->set_input(k_out, layer_prefix + ".k_with_bias");
    k_reshape_4d_sync->set_input(k_reshape_shape_edge, layer_prefix + ".k_reshape_shape");
    k_reshape_4d_sync->set_initial_ids({layer_prefix + ".k_with_bias", layer_prefix + ".k_reshape_shape"});
    pimpl_->graph->add_node(k_reshape_4d_sync);

    k_reshape_4d->set_input(k_reshape_4d_sync->get_output(), "default");
    k_reshape_4d->set_input_names(layer_prefix + ".k_with_bias", layer_prefix + ".k_reshape_shape");
    k_reshape_4d->set_output_name(layer_prefix + ".k_4d");
    k_reshape_4d->set_node_name(layer_prefix + ".k_reshape_4d");
    pimpl_->graph->add_node(k_reshape_4d);
    graph_edge_ptr k_4d = k_reshape_4d->get_output("default");

    auto k_transpose = std::make_shared<transpose_node>();
    k_transpose->set_perm({0, 2, 1, 3});
    k_transpose->set_input(k_4d, "default");
    k_transpose->set_input_name(layer_prefix + ".k_4d");
    k_transpose->set_output_name(layer_prefix + ".k_transposed");
    k_transpose->set_node_name(layer_prefix + ".k_transpose");
    pimpl_->graph->add_node(k_transpose);
    graph_edge_ptr k_transposed = k_transpose->get_output("default");

    auto rope_k = std::make_shared<rope_node>();
    rope_k->set_config(layer_head_dim, pimpl_->max_seq_len, pimpl_->rope_freq_base,
               pimpl_->rope_scaling_factor, pimpl_->rope_scaling_type,
               pimpl_->rope_scaling_orig_ctx);
    
    auto rope_k_sync = std::make_shared<result_message_sync_node>();
    rope_k_sync->set_input(k_transposed, layer_prefix + ".k_transposed");
    rope_k_sync->set_input(position_ids_edge, "position_ids");
    rope_k_sync->set_initial_ids({layer_prefix + ".k_transposed", "position_ids"});
    pimpl_->graph->add_node(rope_k_sync);
    
    rope_k->set_input(rope_k_sync->get_output(), "default");
    rope_k->set_input_names({layer_prefix + ".k_transposed", "position_ids"});
    rope_k->set_output_name(layer_prefix + ".k_rope");
    rope_k->set_node_name(layer_prefix + ".rope_k");
    pimpl_->graph->add_node(rope_k);
    graph_edge_ptr k_rope = rope_k->get_output("default");

    auto k_transpose_back = std::make_shared<transpose_node>();
    k_transpose_back->set_perm({0, 2, 1, 3});
    k_transpose_back->set_input(k_rope, "default");
    k_transpose_back->set_input_name(layer_prefix + ".k_rope");
    k_transpose_back->set_output_name(layer_prefix + ".k_rope_4d");
    k_transpose_back->set_node_name(layer_prefix + ".k_transpose_back");
    pimpl_->graph->add_node(k_transpose_back);
    graph_edge_ptr k_rope_4d = k_transpose_back->get_output("default");

    auto k_reshape_3d = std::make_shared<reshape_node>();
    auto k_reshape_3d_sync = std::make_shared<result_message_sync_node>();
    k_reshape_3d_sync->set_input(k_rope_4d, layer_prefix + ".k_rope_4d");
    k_reshape_3d_sync->set_input(reshape_back_shape_edge, layer_prefix + ".reshape_back_shape");
    k_reshape_3d_sync->set_initial_ids({layer_prefix + ".k_rope_4d", layer_prefix + ".reshape_back_shape"});
    pimpl_->graph->add_node(k_reshape_3d_sync);

    k_reshape_3d->set_input(k_reshape_3d_sync->get_output(), "default");
    k_reshape_3d->set_input_names(layer_prefix + ".k_rope_4d", layer_prefix + ".reshape_back_shape");
    k_reshape_3d->set_output_name(layer_prefix + ".k_rope_out");
    k_reshape_3d->set_node_name(layer_prefix + ".k_reshape_3d");
    pimpl_->graph->add_node(k_reshape_3d);
    graph_edge_ptr k_rope_out = k_reshape_3d->get_output("default");

    // Grouped Query Attention
    auto attn = std::make_shared<grouped_attention_node>();
    std::optional<dynamic_tensor> attn_sinks;
    {
      auto it_sinks = pimpl_->weights.find(layer_prefix + ".attn_sinks.weight");
      if (it_sinks == pimpl_->weights.end()) {
        it_sinks = pimpl_->weights.find(layer_prefix + ".attn_sinks");
      }

      if (it_sinks != pimpl_->weights.end()) {
        const auto& t = it_sinks->second;
        if (t.ndim() != 1 || t.dim(0) != pimpl_->num_q_heads) {
          throw std::runtime_error(layer_prefix + ": attn_sinks shape mismatch (expected [" +
                                   std::to_string(pimpl_->num_q_heads) + "], got [" +
                                   std::to_string(t.dim(0)) + "])");
        }
        attn_sinks = t;
      }
    }
    
    attn->set_config(pimpl_->num_q_heads, pimpl_->num_kv_heads, layer_head_dim,
                     pimpl_->attention_sliding_window, attn_sinks);

    // Store attention node for KV cache management
    pimpl_->attention_nodes.push_back(attn);

    auto attn_sync = std::make_shared<result_message_sync_node>();
    attn_sync->set_input(q_rope_out, layer_prefix + ".q_rope_out");
    attn_sync->set_input(k_rope_out, layer_prefix + ".k_rope_out");
    attn_sync->set_input(v_out, layer_prefix + ".v_with_bias");
    attn_sync->set_initial_ids({layer_prefix + ".q_rope_out", layer_prefix + ".k_rope_out", layer_prefix + ".v_with_bias"});
    pimpl_->graph->add_node(attn_sync);

    attn->set_input(attn_sync->get_output(), "default");
    attn->set_input_names({layer_prefix + ".q_rope_out", layer_prefix + ".k_rope_out", layer_prefix + ".v_with_bias"});
    attn->set_output_name(layer_prefix + ".attn_out");
    attn->set_node_name(layer_prefix + ".grouped_attn");
    pimpl_->graph->add_node(attn);
    graph_edge_ptr attn_out = attn->get_output("default");

    // Attention output projection
    auto attn_proj = std::make_shared<matmul_transpose_mixed_node>();
    auto attn_proj_sync = std::make_shared<result_message_sync_node>();
    attn_proj_sync->set_input(attn_out, layer_prefix + ".attn_out");
    attn_proj_sync->set_input(layer_weights[layer][layer_prefix + ".attn_output.weight"], layer_prefix + ".attn_output.weight");
    attn_proj_sync->set_initial_ids({layer_prefix + ".attn_out", layer_prefix + ".attn_output.weight"});
    pimpl_->graph->add_node(attn_proj_sync);

    attn_proj->set_input(attn_proj_sync->get_output(), "default");
    attn_proj->set_input_names(layer_prefix + ".attn_out", layer_prefix + ".attn_output.weight");
    attn_proj->set_output_name(layer_prefix + ".attn_proj_out");
    attn_proj->set_node_name(layer_prefix + ".attn_proj");
    pimpl_->graph->add_node(attn_proj);
    
    // Attention output bias
    auto attn_bias_add = std::make_shared<add_node>();
    auto attn_bias_sync = std::make_shared<result_message_sync_node>();
    attn_bias_sync->set_input(attn_proj->get_output("default"), layer_prefix + ".attn_proj_out");
    attn_bias_sync->set_input(layer_weights[layer][layer_prefix + ".attn_output.bias"], layer_prefix + ".attn_output.bias");
    attn_bias_sync->set_initial_ids({layer_prefix + ".attn_proj_out", layer_prefix + ".attn_output.bias"});
    pimpl_->graph->add_node(attn_bias_sync);
    
    attn_bias_add->set_input(attn_bias_sync->get_output(), "default");
    attn_bias_add->set_input_names(layer_prefix + ".attn_proj_out", layer_prefix + ".attn_output.bias");
    attn_bias_add->set_output_name(layer_prefix + ".attn_with_bias");
    attn_bias_add->set_node_name(layer_prefix + ".attn_proj_bias");
    pimpl_->graph->add_node(attn_bias_add);
    graph_edge_ptr attn_proj_out = attn_bias_add->get_output("default");

    // 2.6 Attention residual connection
    auto attn_residual = std::make_shared<add_node>();
    auto attn_res_sync = std::make_shared<result_message_sync_node>();
    attn_res_sync->set_input(layer_input, layer_input_name);
    attn_res_sync->set_input(attn_proj_out, layer_prefix + ".attn_with_bias");
    attn_res_sync->set_initial_ids({layer_input_name, layer_prefix + ".attn_with_bias"});
    pimpl_->graph->add_node(attn_res_sync);

    attn_residual->set_input(attn_res_sync->get_output(), "default");
    attn_residual->set_input_names(layer_input_name, layer_prefix + ".attn_with_bias");
    attn_residual->set_output_name(layer_prefix + ".attn_residual_out");
    attn_residual->set_node_name(layer_prefix + ".attn_residual");
    pimpl_->graph->add_node(attn_residual);
    graph_edge_ptr attn_res_out = attn_residual->get_output("default");

    // FFN RMSNorm
    auto ffn_norm = std::make_shared<rmsnorm_node>();
    ffn_norm->set_epsilon(1e-5f);

    auto ffn_norm_sync = std::make_shared<result_message_sync_node>();
    ffn_norm_sync->set_input(attn_res_out, layer_prefix + ".attn_residual_out");
    ffn_norm_sync->set_input(layer_weights[layer][layer_prefix + ".post_attention_norm.weight"], layer_prefix + ".post_attention_norm.weight");
    ffn_norm_sync->set_initial_ids({layer_prefix + ".attn_residual_out", layer_prefix + ".post_attention_norm.weight"});
    pimpl_->graph->add_node(ffn_norm_sync);

    ffn_norm->set_input(ffn_norm_sync->get_output(), "default");
    ffn_norm->set_input_names(layer_prefix + ".attn_residual_out", layer_prefix + ".post_attention_norm.weight");
    ffn_norm->set_output_name(layer_prefix + ".ffn_norm_out");
    ffn_norm->set_node_name(layer_prefix + ".ffn_norm");
    pimpl_->graph->add_node(ffn_norm);
    graph_edge_ptr ffn_norm_out = ffn_norm->get_output("default");

    // MoE Router
    auto router = std::make_shared<moe_router_node>();
    router->set_config(pimpl_->num_experts, pimpl_->expert_top_k);

    auto router_sync = std::make_shared<result_message_sync_node>();
    router_sync->set_input(ffn_norm_out, layer_prefix + ".ffn_norm_out");
    router_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_gate_inp.weight"], layer_prefix + ".ffn_gate_inp.weight");
    router_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_gate_inp.bias"], layer_prefix + ".ffn_gate_inp.bias");
    router_sync->set_initial_ids({layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_gate_inp.weight", layer_prefix + ".ffn_gate_inp.bias"});
    pimpl_->graph->add_node(router_sync);

    router->set_input(router_sync->get_output(), "default");
    router->set_input_names({layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_gate_inp.weight", layer_prefix + ".ffn_gate_inp.bias"});
    router->set_output_name(layer_prefix + ".router_out");
    router->set_node_name(layer_prefix + ".moe_router");
    pimpl_->graph->add_node(router);
    graph_edge_ptr router_out = router->get_output("default");

    // Expert MLPs (receive router_output directly for token-level conditional execution)
    std::vector<graph_edge_ptr> expert_outputs;
    for (int expert_id = 0; expert_id < pimpl_->num_experts; ++expert_id) {
      // Sync for 3D weights/2D biases + router_out
      auto weight_sync = std::make_shared<result_message_sync_node>();
      weight_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_up_exps.weight"], layer_prefix + ".ffn_up_exps.weight");
      weight_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_gate_exps.weight"], layer_prefix + ".ffn_gate_exps.weight");
      weight_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_down_exps.weight"], layer_prefix + ".ffn_down_exps.weight");
      weight_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_up_exps.bias"], layer_prefix + ".ffn_up_exps.bias");
      weight_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_gate_exps.bias"], layer_prefix + ".ffn_gate_exps.bias");
      weight_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_down_exps.bias"], layer_prefix + ".ffn_down_exps.bias");
      weight_sync->set_input(router_out, layer_prefix + ".router_out");
      weight_sync->set_initial_ids({layer_prefix + ".ffn_up_exps.weight", layer_prefix + ".ffn_gate_exps.weight", layer_prefix + ".ffn_down_exps.weight", layer_prefix + ".ffn_up_exps.bias", layer_prefix + ".ffn_gate_exps.bias", layer_prefix + ".ffn_down_exps.bias", layer_prefix + ".router_out"});
      pimpl_->graph->add_node(weight_sync);

      // Slice 3D weights -> 2D/1D views for this expert
      auto slice = std::make_shared<moe_expert_weight_slice_node>(expert_id);
      slice->set_input(weight_sync->get_output(), "default");
      slice->set_input_names({layer_prefix + ".ffn_up_exps.weight", layer_prefix + ".ffn_gate_exps.weight", layer_prefix + ".ffn_down_exps.weight", layer_prefix + ".ffn_up_exps.bias", layer_prefix + ".ffn_gate_exps.bias", layer_prefix + ".ffn_down_exps.bias", layer_prefix + ".router_out"});
      slice->set_output_names({layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_up",
                               layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_gate",
                               layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_down",
                               layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_up",
                               layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_gate",
                               layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_down",
                               layer_prefix + ".expert_" + std::to_string(expert_id) + ".router_out"});
      slice->set_node_name(layer_prefix + ".expert_" + std::to_string(expert_id) + ".slice");
      pimpl_->graph->add_node(slice);
      graph_edge_ptr sliced_weights = slice->get_output("default");

      // Sync for expert MLP: ffn_norm_out + 2D/1D sliced weights
      auto expert_sync = std::make_shared<result_message_sync_node>();
      expert_sync->set_input(ffn_norm_out, layer_prefix + ".ffn_norm_out");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_up");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_gate");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_down");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_up");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_gate");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_down");
      expert_sync->set_input(sliced_weights, layer_prefix + ".expert_" + std::to_string(expert_id) + ".router_out");
      expert_sync->set_initial_ids({layer_prefix + ".ffn_norm_out", 
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_up",
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_gate",
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_down",
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_up",
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_gate",
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_down",
                                     layer_prefix + ".expert_" + std::to_string(expert_id) + ".router_out"});
      pimpl_->graph->add_node(expert_sync);

      // Expert MLP node (now receives 2D/1D weights)
      auto expert = std::make_shared<expert_mlp_node>(expert_id);
      expert->set_input(expert_sync->get_output(), "default");
      expert->set_input_names({layer_prefix + ".ffn_norm_out", layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_up", layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_gate", layer_prefix + ".expert_" + std::to_string(expert_id) + ".w_down", layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_up", layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_gate", layer_prefix + ".expert_" + std::to_string(expert_id) + ".b_down", layer_prefix + ".expert_" + std::to_string(expert_id) + ".router_out"});
      expert->set_output_name(layer_prefix + ".expert_" + std::to_string(expert_id) + "_out");
      expert->set_node_name(layer_prefix + ".expert_" + std::to_string(expert_id));
      pimpl_->graph->add_node(expert);
      expert_outputs.push_back(expert->get_output("default"));
    }

    // Expert merge
    auto expert_merge = std::make_shared<expert_merge_node>();
    expert_merge->set_config(pimpl_->num_experts, pimpl_->expert_top_k);

    std::vector<std::string> merge_input_names = {layer_prefix + ".router_out"};
    for (int i = 0; i < pimpl_->num_experts; ++i) {
      merge_input_names.push_back(layer_prefix + ".expert_" + std::to_string(i) + "_out");
    }

    auto merge_sync = std::make_shared<result_message_sync_node>();
    merge_sync->set_input(router_out, layer_prefix + ".router_out");
    for (int i = 0; i < pimpl_->num_experts; ++i) {
      merge_sync->set_input(expert_outputs[i], layer_prefix + ".expert_" + std::to_string(i) + "_out");
    }
    merge_sync->set_initial_ids(merge_input_names);
    pimpl_->graph->add_node(merge_sync);

    expert_merge->set_input(merge_sync->get_output(), "default");
    expert_merge->set_input_names(merge_input_names);
    expert_merge->set_output_name(layer_prefix + ".expert_merge_out");
    expert_merge->set_node_name(layer_prefix + ".expert_merge");
    pimpl_->graph->add_node(expert_merge);
    graph_edge_ptr merge_out = expert_merge->get_output("default");

    // FFN residual
    auto ffn_residual = std::make_shared<add_node>();
    auto ffn_res_sync = std::make_shared<result_message_sync_node>();
    ffn_res_sync->set_input(attn_res_out, layer_prefix + ".attn_residual_out");
    ffn_res_sync->set_input(merge_out, layer_prefix + ".expert_merge_out");
    ffn_res_sync->set_initial_ids({layer_prefix + ".attn_residual_out", layer_prefix + ".expert_merge_out"});
    pimpl_->graph->add_node(ffn_res_sync);

    ffn_residual->set_input(ffn_res_sync->get_output(), "default");
    ffn_residual->set_input_names(layer_prefix + ".attn_residual_out", layer_prefix + ".expert_merge_out");
    ffn_residual->set_output_name(layer_prefix + ".ffn_residual_out");
    ffn_residual->set_node_name(layer_prefix + ".ffn_residual");
    pimpl_->graph->add_node(ffn_residual);

    current = ffn_residual->get_output("default");
    current_name = layer_prefix + ".ffn_residual_out";
  }

  // Final RMSNorm
  auto final_norm = std::make_shared<rmsnorm_node>();
  final_norm->set_epsilon(1e-5f);

  auto final_norm_sync = std::make_shared<result_message_sync_node>();
  final_norm_sync->set_input(current, current_name);
  final_norm_sync->set_input(output_norm_weight_edge, "output_norm.weight");
  final_norm_sync->set_initial_ids({current_name, "output_norm.weight"});
  pimpl_->graph->add_node(final_norm_sync);

  final_norm->set_input(final_norm_sync->get_output(), "default");
  final_norm->set_input_names(current_name, "output_norm.weight");
  final_norm->set_output_name("final_norm_out");
  final_norm->set_node_name("final_norm");
  pimpl_->graph->add_node(final_norm);
  graph_edge_ptr final_norm_out = final_norm->get_output("default");

  // Output projection
  auto output_proj = std::make_shared<matmul_transpose_mixed_node>();

  auto output_proj_sync = std::make_shared<result_message_sync_node>();
  output_proj_sync->set_input(final_norm_out, "final_norm_out");
  output_proj_sync->set_input(output_weight_edge, "output.weight");
  output_proj_sync->set_initial_ids({"final_norm_out", "output.weight"});
  pimpl_->graph->add_node(output_proj_sync);

  output_proj->set_input(output_proj_sync->get_output(), "default");
  output_proj->set_input_names("final_norm_out", "output.weight");
  output_proj->set_output_name("logits");
  output_proj->set_node_name("output_projection");
  pimpl_->graph->add_node(output_proj);
  graph_edge_ptr logits_edge = output_proj->get_output("default");

  // Output sync
  auto output_sync = std::make_shared<result_message_sync_node>();
  pimpl_->graph->add_node(output_sync);
  output_sync->set_input(logits_edge, "logits");
  output_sync->set_initial_ids({"logits"});

  pimpl_->output_node->set_input(output_sync->get_output(), "default");

  // Add I/O nodes to graph
  pimpl_->graph->add_node(pimpl_->input_node);
  pimpl_->graph->add_node(pimpl_->output_node);

  // Deploy graph
  pimpl_->proc = std::make_unique<graph_proc>();
  pimpl_->proc->deploy(pimpl_->graph);

  // Initialize KV caches after graph is built
  initialize_kv_caches();
}

void gpt_oss_engine::wire_io_nodes(graph_edge_ptr input_placeholder, graph_edge_ptr logits_output) {
  pimpl_->input_node = std::make_shared<model_input_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  auto input_ids_out = extractor->add_output("input_ids");

  for (uint32_t i = 0; i < pimpl_->graph->get_node_count(); ++i) {
    auto node = pimpl_->graph->get_node(i);
    const auto& inputs = node->get_inputs();
    for (const auto& [port_name, input_edge] : inputs) {
      if (input_edge == input_placeholder) {
        node->set_input(input_ids_out, port_name);
      }
    }
  }

  if (!pimpl_->constant_nodes.empty()) {
    for (auto& const_node : pimpl_->constant_nodes) {
      const_node->set_input(input_ids_out, "default");
    }
  }

  auto output_sync = std::make_shared<result_message_sync_node>();
  pimpl_->graph->add_node(output_sync);
  output_sync->set_input(logits_output, "logits");
  output_sync->set_initial_ids({"logits"});

  pimpl_->output_node->set_input(output_sync->get_output(), "default");

  pimpl_->graph->add_node(pimpl_->input_node);
  pimpl_->graph->add_node(pimpl_->output_node);

  pimpl_->proc = std::make_unique<graph_proc>();
  pimpl_->proc->deploy(pimpl_->graph);
}

std::string gpt_oss_engine::generate(const std::string& prompt, size_t max_tokens,
                                     float temperature) {
  if (!pimpl_->loaded) {
    throw std::runtime_error("Model not loaded");
  }

  std::vector<uint32_t> tokens = pimpl_->tokenizer->encode(prompt);

  if (pimpl_->tokenizer->add_bos_token()) {
    const uint32_t bos = pimpl_->tokenizer->bos_token_id();
    if (tokens.empty() || tokens.front() != bos) {
      tokens.insert(tokens.begin(), bos);
    }
  }
  if (pimpl_->tokenizer->add_eos_token()) {
    const uint32_t eos = pimpl_->tokenizer->eos_token_id();
    if (tokens.empty() || tokens.back() != eos) {
      tokens.push_back(eos);
    }
  }

  // Reset KV caches and position counter at the start of generation
  reset_kv_caches();
  pimpl_->cached_position_ = 0;

  std::string generated_text;
  std::unordered_map<std::string, dynamic_tensor> outputs;
  bool output_received = false;

  pimpl_->output_node->set_callback(
      [&outputs, &output_received](const std::unordered_map<std::string, dynamic_tensor>& result) {
        outputs = result;
        output_received = true;
      });

  for (size_t step = 0; step < max_tokens; ++step) {
    output_received = false;

    std::vector<int64_t> shape;
    dynamic_tensor input_tensor;
    dynamic_tensor position_ids_tensor;

    if (step == 0) {
      // ===== Prefill phase: process all prompt tokens =====
      shape = {1, static_cast<int64_t>(tokens.size())};
      input_tensor = dynamic_tensor(dtype::int32, shape);
      int32_t* data = input_tensor.data_ptr<int32_t>();
      for (size_t j = 0; j < tokens.size(); ++j) {
        data[j] = static_cast<int32_t>(tokens[j]);
      }
      
      // Generate position_ids: [0, 1, 2, ..., seq_len-1]
      position_ids_tensor = dynamic_tensor(dtype::int64, {static_cast<int64_t>(tokens.size())});
      int64_t* pos_data = position_ids_tensor.data_ptr<int64_t>();
      for (size_t j = 0; j < tokens.size(); ++j) {
        pos_data[j] = static_cast<int64_t>(j);
      }
      pimpl_->cached_position_ = tokens.size();
    } else {
      // ===== Decode phase: process only the last token =====
      shape = {1, 1};
      input_tensor = dynamic_tensor(dtype::int32, shape);
      int32_t* data = input_tensor.data_ptr<int32_t>();
      data[0] = static_cast<int32_t>(tokens.back());
      
      // Generate position_ids: [cached_position_] (current absolute position)
      position_ids_tensor = dynamic_tensor(dtype::int64, {1});
      int64_t* pos_data = position_ids_tensor.data_ptr<int64_t>();
      pos_data[0] = pimpl_->cached_position_;
      pimpl_->cached_position_++;
    }

    pimpl_->input_node->set_tensor("input_ids", input_tensor);
    pimpl_->input_node->set_tensor("position_ids", position_ids_tensor);
    pimpl_->input_node->set_frame_number(step + 1);

    pimpl_->proc->run();

    if (!output_received) {
      throw std::runtime_error("No output received for step " + std::to_string(step));
    }

    auto it = outputs.find("logits");
    if (it == outputs.end()) {
      throw std::runtime_error("No logits output");
    }

    const auto& logits = it->second;
    if (logits.get_dtype() != dtype::float32) {
      throw std::runtime_error("logits must be float32");
    }
    if (logits.ndim() != 3 || logits.dim(0) != 1) {
      throw std::runtime_error("logits must have shape [1, seq_len, vocab_size]");
    }
    const float* logits_data = logits.data_ptr<float>();

    int64_t seq_len = logits.dim(1);
    int64_t vocab_size = logits.dim(2);
    int64_t last_pos = seq_len - 1;
    const float* last_logits = logits_data + (0 * seq_len + last_pos) * vocab_size;

    uint32_t next_token = sample_token(last_logits, vocab_size, temperature);

    if (next_token == pimpl_->tokenizer->eos_token_id()) {
      break;
    }

    tokens.push_back(next_token);

    std::string piece = pimpl_->tokenizer->decode({next_token});
    generated_text += piece;
  }

  return generated_text;
}

uint32_t gpt_oss_engine::sample_token(const float* logits, int64_t vocab_size, float temperature) {
  if (temperature < 1e-6f) {
    return static_cast<uint32_t>(
        std::distance(logits, std::max_element(logits, logits + vocab_size)));
  }

  std::vector<float> probs(vocab_size);
  float max_logit = *std::max_element(logits, logits + vocab_size);

  float sum = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) / temperature);
    sum += probs[i];
  }

  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] /= sum;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probs.begin(), probs.end());

  return static_cast<uint32_t>(dist(gen));
}

bool gpt_oss_engine::is_loaded() const { return pimpl_->loaded; }

int64_t gpt_oss_engine::get_vocab_size() const { return pimpl_->vocab_size; }

int64_t gpt_oss_engine::get_num_layers() const { return pimpl_->num_layers; }

int64_t gpt_oss_engine::get_hidden_dim() const { return pimpl_->hidden_dim; }

void gpt_oss_engine::initialize_kv_caches() {
  if (pimpl_->attention_nodes.empty()) {
    return;
  }
  
  // Cache shape: [batch=1, num_kv_heads, max_seq_len, head_dim]
  int64_t batch = 1;
  int64_t max_seq_len = std::min(pimpl_->kv_cache_size, pimpl_->max_seq_len);
  
  size_t cache_size_mb = 0;
  
  for (auto& attn_node : pimpl_->attention_nodes) {
    // Allocate K and V caches
    dynamic_tensor k_cache(dtype::float32, {batch, pimpl_->num_kv_heads, max_seq_len, pimpl_->head_dim});
    dynamic_tensor v_cache(dtype::float32, {batch, pimpl_->num_kv_heads, max_seq_len, pimpl_->head_dim});
    
    // Initialize to zero
    std::memset(k_cache.data_ptr<float>(), 0, k_cache.bytes());
    std::memset(v_cache.data_ptr<float>(), 0, v_cache.bytes());
    
    cache_size_mb += k_cache.bytes() + v_cache.bytes();
    
    attn_node->set_k_cache(k_cache);
    attn_node->set_v_cache(v_cache);
  }
}

void gpt_oss_engine::reset_kv_caches() {
  for (auto& attn_node : pimpl_->attention_nodes) {
    attn_node->reset_cache();
  }
}

}  // namespace coalsack
