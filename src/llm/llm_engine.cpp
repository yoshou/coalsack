#include "coalsack/llm/llm_engine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>

#include "coalsack/core/graph_proc.h"
#include "coalsack/gguf/gguf_dequant.h"
#include "coalsack/gguf/gguf_multi_loader.h"
#include "coalsack/llm/graph_builder.h"
#include "coalsack/llm/moe_weight_provider.h"
#include "coalsack/nn/model_io_nodes.h"
#include "coalsack/nn/nn_nodes.h"
#include "coalsack/nn/nn_ops/add_node.h"
#include "coalsack/nn/nn_ops/constant_node.h"
#include "coalsack/nn/nn_ops/dense_ffn_node.h"
#include "coalsack/nn/nn_ops/grouped_attention_node.h"
#include "coalsack/nn/nn_ops/matmul_transpose_mixed_node.h"
#include "coalsack/nn/nn_ops/reshape_node.h"
#include "coalsack/nn/result_message_nodes.h"

namespace coalsack {

struct llm_engine::impl {
  // Config
  config cfg;

  // Components
  std::shared_ptr<gguf_multi_loader> loader;
  std::shared_ptr<subgraph> graph;
  std::unique_ptr<graph_proc> proc;

  // I/O nodes
  std::shared_ptr<model_source_node> input_node;
  std::shared_ptr<model_output_node> output_node;

  // Constant nodes (for weight tensors, need frame source connection)
  std::vector<std::shared_ptr<constant_node>> constant_nodes;

  // Attention nodes (for KV cache management)
  std::vector<std::shared_ptr<grouped_attention_node>> attention_nodes;

  // MoE weight providers (one per layer, for on-demand expert weight loading)
  std::vector<std::shared_ptr<moe_weight_provider>> moe_providers;

  // Position tracking for RoPE
  int64_t cached_position_ = 0;
  uint64_t step_counter_ = 0;

  // Model config (from GGUF metadata)
  int64_t num_layers = 24;
  int64_t hidden_dim = 2880;
  int64_t num_q_heads = 64;
  int64_t num_kv_heads = 8;
  int64_t head_dim = 45;  // hidden_dim / num_q_heads
  int64_t num_experts = 32;
  int64_t expert_top_k = 4;
  int64_t expert_ffn_dim = 2880;
  int64_t max_seq_len = 8192;
  int64_t kv_cache_size = std::numeric_limits<int64_t>::max();

  // Attention config
  int64_t attention_sliding_window = 0;

  // RoPE config
  float rope_freq_base = 150000.0f;
  float rope_scaling_factor = 1.0f;
  std::string rope_scaling_type = "none";
  int64_t rope_scaling_orig_ctx = 0;

  // Architecture config (determined from GGUF general.architecture)
  std::string arch_prefix = "gpt-oss";  // "gpt-oss" or "llama4"
  bool has_shared_expert = false;       // gpt-oss: false, llama4: true
  bool has_expert_bias = true;          // gpt-oss: true, llama4: false
  std::string ffn_norm_weight_name = "post_attention_norm.weight";
  bool use_sigmoid_gating = false;    // llama4: true
  bool weight_before_ffn = false;     // llama4: true
  bool use_kq_norm = false;           // llama4: true
  bool use_norm_rope = false;         // llama4: true (NORM adjacent-pair RoPE)
  int64_t no_rope_layer_step = 0;     // llama4 iswa: 4
  int64_t attn_temp_floor_scale = 0;  // llama4 iswa: 8192
  float attn_temp_scale = 0.0f;       // llama4 iswa: 0.1
  float attn_temp_offset = 0.0f;      // llama4 iswa: 1.0

  // Weights (loaded from GGUF)
  std::unordered_map<std::string, dynamic_tensor> weights;

  // Hidden-layer capture config
  std::vector<int> hidden_layer_indices;

  // Per-step output state
  std::vector<float> current_logits_;
  std::unordered_map<int, std::vector<float>> current_hidden_layers_;

  bool loaded = false;
};

llm_engine::llm_engine() : pimpl_(std::make_unique<impl>()) {}

llm_engine::llm_engine(const config& cfg) : pimpl_(std::make_unique<impl>()) {
  pimpl_->cfg = cfg;
  pimpl_->kv_cache_size = cfg.kv_cache_size;
  pimpl_->hidden_layer_indices = cfg.hidden_layer_indices;
}

llm_engine::~llm_engine() = default;

void llm_engine::load(std::shared_ptr<gguf_multi_loader> loader) {
  if (pimpl_->loaded) {
    throw std::runtime_error("Model already loaded");
  }
  if (!loader) {
    throw std::runtime_error("loader is null");
  }

  pimpl_->loader = loader;

  load_config_from_gguf();

  load_weights_from_gguf();

  build_transformer_graph();

  pimpl_->loaded = true;
}

void llm_engine::load_config_from_gguf() {
  auto& loader = *pimpl_->loader;

  // Detect architecture from GGUF general.architecture
  std::string arch = "gpt-oss";
  if (auto v = loader.get_string("general.architecture")) {
    arch = *v;
  }
  pimpl_->arch_prefix = arch;
  spdlog::info("Detected architecture: {}", arch);

  // Architecture-specific defaults
  if (arch == "llama4") {
    pimpl_->has_shared_expert = true;
    pimpl_->has_expert_bias = false;
    pimpl_->ffn_norm_weight_name = "ffn_norm.weight";
    pimpl_->rope_scaling_factor = 1.0f;
    pimpl_->rope_scaling_type = "none";
    pimpl_->use_sigmoid_gating = true;
    pimpl_->weight_before_ffn = true;
    pimpl_->use_kq_norm = true;
    pimpl_->use_norm_rope = true;
  } else {
    pimpl_->has_shared_expert = false;
    pimpl_->has_expert_bias = true;
    pimpl_->ffn_norm_weight_name = "post_attention_norm.weight";
  }

  const std::string& pfx = pimpl_->arch_prefix;

  // Helper for required parameters
  auto get_required_uint32 = [&](const std::string& suffix) -> uint32_t {
    std::string key = pfx + "." + suffix;
    if (auto v = loader.get_uint32(key)) return *v;
    throw std::runtime_error("Missing required parameter: " + key);
  };

  auto get_optional_uint32 = [&](const std::string& suffix) -> std::optional<uint32_t> {
    return loader.get_uint32(pfx + "." + suffix);
  };

  auto get_optional_float = [&](const std::string& suffix) -> std::optional<float> {
    return loader.get_float32(pfx + "." + suffix);
  };

  auto get_optional_string = [&](const std::string& suffix) -> std::optional<std::string> {
    return loader.get_string(pfx + "." + suffix);
  };

  pimpl_->num_layers = get_required_uint32("block_count");
  pimpl_->hidden_dim = get_required_uint32("embedding_length");
  pimpl_->num_q_heads = get_required_uint32("attention.head_count");
  pimpl_->num_kv_heads = get_required_uint32("attention.head_count_kv");

  pimpl_->max_seq_len = get_required_uint32("context_length");

  if (auto v = get_optional_uint32("attention.sliding_window")) {
    pimpl_->attention_sliding_window = static_cast<int64_t>(*v);
    if (arch == "llama4" && *v == 0) {
      pimpl_->no_rope_layer_step = 0;
      pimpl_->attn_temp_floor_scale = 0;
      pimpl_->attn_temp_scale = 0.0f;
      pimpl_->attn_temp_offset = 0.0f;
    }
  } else if (arch == "llama4") {
    // ISWA: no sliding_window key -> every 4th layer is full-attention without RoPE
    pimpl_->attention_sliding_window = 8192;
    pimpl_->no_rope_layer_step = 4;
    pimpl_->attn_temp_floor_scale = 8192;
    pimpl_->attn_temp_scale = 0.1f;
    pimpl_->attn_temp_offset = 1.0f;
    spdlog::info("no attention.sliding_window found -> ISWA mode, no_rope_layer_step=4");
  }

  if (auto v = get_optional_uint32("attention.temperature_length")) {
    pimpl_->attn_temp_floor_scale = static_cast<int64_t>(*v);
  }
  if (auto v = get_optional_float("attention.temperature_scale")) {
    pimpl_->attn_temp_scale = *v;
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
  if (auto v = get_optional_float("rope.freq_base")) {
    pimpl_->rope_freq_base = *v;
  } else {
    throw std::runtime_error("Missing required parameter: " + pfx + ".rope.freq_base");
  }
  // rope.scaling.* is optional
  if (auto v = get_optional_float("rope.scaling.factor")) {
    pimpl_->rope_scaling_factor = *v;
  }
  if (auto v = get_optional_string("rope.scaling.type")) {
    pimpl_->rope_scaling_type = *v;
  }
  if (auto v = get_optional_uint32("rope.scaling.original_context_length")) {
    pimpl_->rope_scaling_orig_ctx = static_cast<int64_t>(*v);
  }
}

void llm_engine::load_weights_from_gguf() {
  auto& loader = *pimpl_->loader;
  const auto& tensor_names = loader.get_tensor_names();
  const auto& shard_paths = loader.get_shard_paths();

  // Initialize MoE weight providers (one per layer) for on-demand loading
  pimpl_->moe_providers.clear();
  for (int64_t layer = 0; layer < pimpl_->num_layers; ++layer) {
    pimpl_->moe_providers.push_back(std::make_shared<moe_weight_provider>(
        pimpl_->loader, shard_paths, pimpl_->cfg.moe_cache_size_bytes));
  }

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
  size_t skipped_moe_count = 0;

  for (const auto& name : tensor_names) {
    // Skip MoE expert weights (loaded on-demand via moe_weight_provider)
    if (name.find("_exps.weight") != std::string::npos ||
        name.find("_exps.bias") != std::string::npos) {
      ++skipped_moe_count;
      continue;
    }
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

    dtype output_dtype = (info.type == ggml_type::F32) ? dtype::float32 : dtype::float16;

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
      // Dequantize to F32 temp buffer, then convert to F16 to save memory
      std::vector<float> tmp_f32(numel);
      if (!dequantize_tensor(raw_data.data(), tmp_f32.data(), numel, info.type)) {
        throw std::runtime_error("Unsupported quantization type for: " + name);
      }
      uint16_t* dst = tensor.data_ptr<uint16_t>();
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = fp32_to_fp16(tmp_f32[i]);
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
}

void llm_engine::build_transformer_graph() {
  pimpl_->graph = std::make_shared<subgraph>();

  // Create I/O nodes
  pimpl_->input_node = std::make_shared<model_source_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  // Build list of required weights (architecture-dependent)
  // Biases are optional — presence is checked at graph construction time
  std::vector<std::string> required_weights = {
      "token_embd.weight",
      "output_norm.weight",
      "output.weight",
  };

  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_prefix = "blk." + std::to_string(layer);
    required_weights.push_back(layer_prefix + ".attn_norm.weight");
    required_weights.push_back(layer_prefix + ".attn_q.weight");
    required_weights.push_back(layer_prefix + ".attn_k.weight");
    required_weights.push_back(layer_prefix + ".attn_v.weight");
    required_weights.push_back(layer_prefix + ".attn_output.weight");
    required_weights.push_back(layer_prefix + "." + pimpl_->ffn_norm_weight_name);
    required_weights.push_back(layer_prefix + ".ffn_gate_inp.weight");
    // shared expert weights
    if (pimpl_->has_shared_expert) {
      required_weights.push_back(layer_prefix + ".ffn_gate_shexp.weight");
      required_weights.push_back(layer_prefix + ".ffn_up_shexp.weight");
      required_weights.push_back(layer_prefix + ".ffn_down_shexp.weight");
    }
  }

  for (const auto& name : required_weights) {
    auto it = pimpl_->weights.find(name);
    if (it == pimpl_->weights.end()) {
      throw std::runtime_error("Required weight not found: " + name);
    }
    pimpl_->input_node->set_tensor(name, it->second);
  }

  if (pimpl_->arch_prefix == "llama4") {
    auto it = pimpl_->weights.find("rope_freqs.weight");
    if (it == pimpl_->weights.end()) {
      throw std::runtime_error("Required weight not found: rope_freqs.weight");
    }
    pimpl_->input_node->set_tensor("rope_freqs.weight", it->second);
  }

  // Register optional weights present in this model
  auto register_optional = [&](const std::string& name) {
    auto it = pimpl_->weights.find(name);
    if (it != pimpl_->weights.end()) {
      pimpl_->input_node->set_tensor(name, it->second);
    }
  };

  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_prefix = "blk." + std::to_string(layer);
    register_optional(layer_prefix + ".attn_q.bias");
    register_optional(layer_prefix + ".attn_k.bias");
    register_optional(layer_prefix + ".attn_v.bias");
    register_optional(layer_prefix + ".attn_output.bias");
    register_optional(layer_prefix + ".ffn_gate_inp.bias");
  }

  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  auto input_ids_edge = extractor->add_output("input_ids");
  auto position_ids_edge = extractor->add_output("position_ids");  // Add position_ids
  graph_edge_ptr rope_freqs_edge;
  if (pimpl_->weights.find("rope_freqs.weight") != pimpl_->weights.end()) {
    rope_freqs_edge = extractor->add_output("rope_freqs.weight");
  }
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
        layer_prefix + ".attn_norm.weight",      layer_prefix + ".attn_q.weight",
        layer_prefix + ".attn_q.bias",           layer_prefix + ".attn_k.weight",
        layer_prefix + ".attn_k.bias",           layer_prefix + ".attn_v.weight",
        layer_prefix + ".attn_v.bias",           layer_prefix + ".attn_output.weight",
        layer_prefix + ".attn_output.bias",      layer_prefix + "." + pimpl_->ffn_norm_weight_name,
        layer_prefix + ".ffn_gate_inp.weight",   layer_prefix + ".ffn_gate_inp.bias",
        layer_prefix + ".ffn_gate_shexp.weight", layer_prefix + ".ffn_up_shexp.weight",
        layer_prefix + ".ffn_down_shexp.weight",
    };

    for (const auto& name : layer_weight_names) {
      if (pimpl_->weights.find(name) != pimpl_->weights.end()) {
        layer_weights[layer][name] = extractor->add_output(name);
      }
    }
  }

  // Graph builder — encapsulates all sync+node pair creation patterns
  graph_builder::config gb_cfg;
  gb_cfg.max_seq_len = pimpl_->max_seq_len;
  gb_cfg.rope_freq_base = pimpl_->rope_freq_base;
  gb_cfg.rope_scaling_factor = pimpl_->rope_scaling_factor;
  gb_cfg.rope_scaling_type = pimpl_->rope_scaling_type;
  gb_cfg.rope_scaling_orig_ctx = pimpl_->rope_scaling_orig_ctx;
  gb_cfg.use_norm_rope = pimpl_->use_norm_rope;
  gb_cfg.num_experts = pimpl_->num_experts;
  gb_cfg.expert_top_k = pimpl_->expert_top_k;
  gb_cfg.use_sigmoid_gating = pimpl_->use_sigmoid_gating;
  gb_cfg.weight_before_ffn = pimpl_->weight_before_ffn;
  gb_cfg.attn_temp_floor_scale = pimpl_->attn_temp_floor_scale;
  gb_cfg.attn_temp_scale = pimpl_->attn_temp_scale;
  gb_cfg.attn_temp_offset = pimpl_->attn_temp_offset;
  graph_builder gb(pimpl_->graph, gb_cfg);

  // Compute attn_scale [1,1,seq_len,1] from position_ids inside the graph.
  // Returns nullptr when temperature scaling is disabled.
  graph_edge_ptr attn_scale_edge =
      gb.make_attn_scale_from_pos_ids(position_ids_edge, "attn_scale", token_embd_weight_edge_raw);

  // Embedding layer
  graph_edge_ptr current = gb.make_embedding(input_ids_edge, "input_ids", token_embd_weight_edge,
                                             "token_embd.weight", "embeddings", "embedding_lookup");
  std::string current_name = "embeddings";

  // Edges to capture per layer for speculative decoding (filled inside the loop)
  std::unordered_map<int, graph_edge_ptr> captured_hidden_edges;

  // Transformer layers
  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    // Insert layer_scheduler_node at layer entrance to break call stack and trim memory
    current = gb.make_layer_scheduler(current);

    std::string layer_str = std::to_string(layer);
    std::string layer_prefix = "blk." + layer_str;
    int64_t layer_head_dim = pimpl_->head_dim;
    graph_edge_ptr layer_input = current;
    std::string layer_input_name = current_name;

    // Input RMSNorm
    graph_edge_ptr norm_out = gb.make_rmsnorm(
        current, current_name, layer_weights[layer][layer_prefix + ".attn_norm.weight"],
        layer_prefix + ".attn_norm.weight", layer_prefix + ".attn_norm_out",
        layer_prefix + ".attn_norm");

    // Q projection (with optional bias)
    auto [q_out, q_out_name] = gb.apply_optional_bias(
        gb.make_matmul(norm_out, layer_prefix + ".attn_norm_out",
                       layer_weights[layer][layer_prefix + ".attn_q.weight"],
                       layer_prefix + ".attn_q.weight", layer_prefix + ".q_proj_out",
                       layer_prefix + ".q_proj"),
        layer_prefix + ".q_proj_out", layer_weights[layer], layer_prefix + ".attn_q.bias",
        layer_prefix + ".q_with_bias", layer_prefix + ".q_bias");

    // K projection (with optional bias)
    auto [k_out, k_out_name] = gb.apply_optional_bias(
        gb.make_matmul(norm_out, layer_prefix + ".attn_norm_out",
                       layer_weights[layer][layer_prefix + ".attn_k.weight"],
                       layer_prefix + ".attn_k.weight", layer_prefix + ".k_proj_out",
                       layer_prefix + ".k_proj"),
        layer_prefix + ".k_proj_out", layer_weights[layer], layer_prefix + ".attn_k.bias",
        layer_prefix + ".k_with_bias", layer_prefix + ".k_bias");

    // V projection (with optional bias)
    auto [v_out, v_out_name] = gb.apply_optional_bias(
        gb.make_matmul(norm_out, layer_prefix + ".attn_norm_out",
                       layer_weights[layer][layer_prefix + ".attn_v.weight"],
                       layer_prefix + ".attn_v.weight", layer_prefix + ".v_proj_out",
                       layer_prefix + ".v_proj"),
        layer_prefix + ".v_proj_out", layer_weights[layer], layer_prefix + ".attn_v.bias",
        layer_prefix + ".v_with_bias", layer_prefix + ".v_bias");

    // 2.3 Reshape Q/K for RoPE (3D → 4D → transpose → RoPE → transpose → 4D → 3D)

    // Shape constant tensors for reshape operations
    // Q: [batch, seq, hidden] → [batch, seq, num_q_heads, head_dim]
    dynamic_tensor q_reshape_shape(dtype::int64, {4});
    {
      auto d = q_reshape_shape.data_ptr<int64_t>();
      d[0] = 0;
      d[1] = 0;
      d[2] = pimpl_->num_q_heads;
      d[3] = -1;
    }
    auto q_reshape_shape_edge =
        gb.make_shape_const(q_reshape_shape, layer_prefix + ".q_reshape_shape", q_out);

    // K: [batch, seq, hidden] → [batch, seq, num_kv_heads, head_dim]
    dynamic_tensor k_reshape_shape(dtype::int64, {4});
    {
      auto d = k_reshape_shape.data_ptr<int64_t>();
      d[0] = 0;
      d[1] = 0;
      d[2] = pimpl_->num_kv_heads;
      d[3] = -1;
    }
    auto k_reshape_shape_edge =
        gb.make_shape_const(k_reshape_shape, layer_prefix + ".k_reshape_shape", k_out);

    // Back: [batch, seq, heads * head_dim]
    dynamic_tensor reshape_back_shape(dtype::int64, {3});
    {
      auto d = reshape_back_shape.data_ptr<int64_t>();
      d[0] = 0;
      d[1] = 0;
      d[2] = -1;
    }
    auto reshape_back_shape_edge =
        gb.make_shape_const(reshape_back_shape, layer_prefix + ".reshape_back_shape", q_out);

    // Process Q: reshape → transpose → [kq_norm] → [RoPE / attn_temp] → transpose-back →
    // reshape-back
    graph_edge_ptr q_4d =
        gb.make_reshape(q_out, q_out_name, q_reshape_shape_edge, layer_prefix + ".q_reshape_shape",
                        layer_prefix + ".q_4d", layer_prefix + ".q_reshape_4d");
    graph_edge_ptr q_transposed =
        gb.make_transpose(q_4d, layer_prefix + ".q_4d", {0, 2, 1, 3},
                          layer_prefix + ".q_transposed", layer_prefix + ".q_transpose");

    // Determine if this layer skips RoPE (ISWA: every 4th layer is full-attention)
    const bool is_no_rope_layer =
        pimpl_->no_rope_layer_step > 0 && (layer + 1) % pimpl_->no_rope_layer_step == 0;

    graph_edge_ptr q_rope;
    std::string q_rope_edge_name;
    if (is_no_rope_layer) {
      // No-rope layer: skip positional encoding, apply optional Q temperature scaling
      if (attn_scale_edge) {
        q_rope =
            gb.make_mul(q_transposed, layer_prefix + ".q_transposed", attn_scale_edge, "attn_scale",
                        layer_prefix + ".q_attn_temp_scaled", layer_prefix + ".q_attn_temp_mul");
        q_rope_edge_name = layer_prefix + ".q_attn_temp_scaled";
      } else {
        q_rope = q_transposed;
        q_rope_edge_name = layer_prefix + ".q_transposed";
      }
    } else {
      // Optional KQ L2 norm before RoPE
      graph_edge_ptr q_to_rope = q_transposed;
      std::string q_to_rope_name = layer_prefix + ".q_transposed";
      if (pimpl_->use_kq_norm) {
        q_to_rope = gb.make_l2norm(q_transposed, layer_prefix + ".q_transposed",
                                   layer_prefix + ".q_normed", layer_prefix + ".kq_norm_q");
        q_to_rope_name = layer_prefix + ".q_normed";
      }
      q_rope = gb.make_rope(q_to_rope, q_to_rope_name, position_ids_edge, rope_freqs_edge,
                            layer_head_dim, layer_prefix + ".q_rope", layer_prefix + ".rope_q");
      q_rope_edge_name = layer_prefix + ".q_rope";
    }

    graph_edge_ptr q_rope_4d =
        gb.make_transpose(q_rope, q_rope_edge_name, {0, 2, 1, 3}, layer_prefix + ".q_rope_4d",
                          layer_prefix + ".q_transpose_back");
    graph_edge_ptr q_rope_out =
        gb.make_reshape(q_rope_4d, layer_prefix + ".q_rope_4d", reshape_back_shape_edge,
                        layer_prefix + ".reshape_back_shape", layer_prefix + ".q_rope_out",
                        layer_prefix + ".q_reshape_3d");

    // Process K: reshape → transpose → [kq_norm] → [RoPE] → transpose-back → reshape-back
    graph_edge_ptr k_4d =
        gb.make_reshape(k_out, k_out_name, k_reshape_shape_edge, layer_prefix + ".k_reshape_shape",
                        layer_prefix + ".k_4d", layer_prefix + ".k_reshape_4d");
    graph_edge_ptr k_transposed =
        gb.make_transpose(k_4d, layer_prefix + ".k_4d", {0, 2, 1, 3},
                          layer_prefix + ".k_transposed", layer_prefix + ".k_transpose");

    graph_edge_ptr k_rope;
    std::string k_rope_edge_name;
    if (is_no_rope_layer) {
      // No-rope layer: pass K directly
      k_rope = k_transposed;
      k_rope_edge_name = layer_prefix + ".k_transposed";
    } else {
      // Optional KQ L2 norm before RoPE
      graph_edge_ptr k_to_rope = k_transposed;
      std::string k_to_rope_name = layer_prefix + ".k_transposed";
      if (pimpl_->use_kq_norm) {
        k_to_rope = gb.make_l2norm(k_transposed, layer_prefix + ".k_transposed",
                                   layer_prefix + ".k_normed", layer_prefix + ".kq_norm_k");
        k_to_rope_name = layer_prefix + ".k_normed";
      }
      k_rope = gb.make_rope(k_to_rope, k_to_rope_name, position_ids_edge, rope_freqs_edge,
                            layer_head_dim, layer_prefix + ".k_rope", layer_prefix + ".rope_k");
      k_rope_edge_name = layer_prefix + ".k_rope";
    }

    graph_edge_ptr k_rope_4d =
        gb.make_transpose(k_rope, k_rope_edge_name, {0, 2, 1, 3}, layer_prefix + ".k_rope_4d",
                          layer_prefix + ".k_transpose_back");
    graph_edge_ptr k_rope_out =
        gb.make_reshape(k_rope_4d, layer_prefix + ".k_rope_4d", reshape_back_shape_edge,
                        layer_prefix + ".reshape_back_shape", layer_prefix + ".k_rope_out",
                        layer_prefix + ".k_reshape_3d");

    // Grouped Query Attention
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
    int64_t layer_sliding_window = pimpl_->attention_sliding_window;
    if (pimpl_->no_rope_layer_step > 0) {
      // ISWA pattern: no-rope layers use full attention, others use chunked sliding window
      layer_sliding_window = is_no_rope_layer ? 0 : pimpl_->attention_sliding_window;
    }
    graph_edge_ptr attn_out = gb.make_grouped_attn(
        q_rope_out, layer_prefix + ".q_rope_out", k_rope_out, layer_prefix + ".k_rope_out", v_out,
        v_out_name, pimpl_->num_q_heads, pimpl_->num_kv_heads, layer_head_dim, layer_sliding_window,
        attn_sinks, layer_prefix + ".attn_out", layer_prefix + ".grouped_attn");

    // Attention output projection (with optional bias)
    auto [attn_proj_out, attn_proj_out_name] = gb.apply_optional_bias(
        gb.make_matmul(attn_out, layer_prefix + ".attn_out",
                       layer_weights[layer][layer_prefix + ".attn_output.weight"],
                       layer_prefix + ".attn_output.weight", layer_prefix + ".attn_proj_out",
                       layer_prefix + ".attn_proj"),
        layer_prefix + ".attn_proj_out", layer_weights[layer], layer_prefix + ".attn_output.bias",
        layer_prefix + ".attn_with_bias", layer_prefix + ".attn_proj_bias");

    // Attention residual connection
    graph_edge_ptr attn_res_out =
        gb.make_add(layer_input, layer_input_name, attn_proj_out, attn_proj_out_name,
                    layer_prefix + ".attn_residual_out", layer_prefix + ".attn_residual");

    // FFN RMSNorm
    const std::string ffn_norm_weight_key = layer_prefix + "." + pimpl_->ffn_norm_weight_name;
    graph_edge_ptr ffn_norm_out =
        gb.make_rmsnorm(attn_res_out, layer_prefix + ".attn_residual_out",
                        layer_weights[layer][ffn_norm_weight_key], ffn_norm_weight_key,
                        layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_norm");

    // MoE Router (weight + optional bias)
    const bool has_router_bias =
        layer_weights[layer].count(layer_prefix + ".ffn_gate_inp.bias") > 0;
    graph_edge_ptr router_out = gb.make_moe_router(
        ffn_norm_out, layer_prefix + ".ffn_norm_out",
        layer_weights[layer][layer_prefix + ".ffn_gate_inp.weight"],
        layer_prefix + ".ffn_gate_inp.weight",
        has_router_bias ? layer_weights[layer].at(layer_prefix + ".ffn_gate_inp.bias") : nullptr,
        layer_prefix + ".ffn_gate_inp.bias", layer_prefix + ".router_out",
        layer_prefix + ".moe_router");

    // On-demand MoE weight fetch node (layer-level, loads expert weights from disk)
    auto fetch_outputs = gb.make_moe_weight_fetch(pimpl_->moe_providers[layer], layer_prefix,
                                                  router_out, pimpl_->has_expert_bias);

    // Expert MLPs (receive dynamically loaded weights from fetch node)
    const bool has_expert_bias = pimpl_->has_expert_bias;
    const auto expert_act_type = (pimpl_->arch_prefix == "gpt-oss")
                                     ? expert_mlp_node::activation_type::MODIFIED_SWIGLU
                                     : expert_mlp_node::activation_type::STANDARD_SWIGLU;
    std::vector<graph_edge_ptr> expert_outputs;
    for (int expert_id = 0; expert_id < pimpl_->num_experts; ++expert_id) {
      const std::string ep = layer_prefix + ".expert_" + std::to_string(expert_id);
      expert_outputs.push_back(
          gb.make_expert_mlp(expert_id, ffn_norm_out, layer_prefix + ".ffn_norm_out",
                             fetch_outputs[expert_id], ep, has_expert_bias, expert_act_type));
    }

    // Expert merge
    graph_edge_ptr merge_out =
        gb.make_expert_merge(router_out, layer_prefix + ".router_out", expert_outputs, layer_prefix,
                             layer_prefix + ".expert_merge_out");
    std::string merge_out_name = layer_prefix + ".expert_merge_out";

    // Shared Expert FFN (optional, Llama-4 style)
    const std::string shexp_gate_name = layer_prefix + ".ffn_gate_shexp.weight";
    const std::string shexp_up_name = layer_prefix + ".ffn_up_shexp.weight";
    const std::string shexp_down_name = layer_prefix + ".ffn_down_shexp.weight";
    if (pimpl_->has_shared_expert && layer_weights[layer].count(shexp_gate_name) &&
        layer_weights[layer].count(shexp_up_name) && layer_weights[layer].count(shexp_down_name)) {
      // Shared Expert FFN (dense_ffn_node)
      graph_edge_ptr shexp_out = gb.make_dense_ffn(
          ffn_norm_out, layer_prefix + ".ffn_norm_out", layer_weights[layer][shexp_gate_name],
          shexp_gate_name, layer_weights[layer][shexp_up_name], shexp_up_name,
          layer_weights[layer][shexp_down_name], shexp_down_name, layer_prefix + ".shexp_out",
          layer_prefix + ".shexp_mlp");
      // Add shared expert output to MoE merge output
      merge_out = gb.make_add(merge_out, merge_out_name, shexp_out, layer_prefix + ".shexp_out",
                              layer_prefix + ".moe_combined_out", layer_prefix + ".shexp_add");
      merge_out_name = layer_prefix + ".moe_combined_out";
    }

    // FFN residual
    current =
        gb.make_add(attn_res_out, layer_prefix + ".attn_residual_out", merge_out, merge_out_name,
                    layer_prefix + ".ffn_residual_out", layer_prefix + ".ffn_residual");
    current_name = layer_prefix + ".ffn_residual_out";

    // Capture hidden state for speculative decoding if this layer is configured
    {
      const auto& hl = pimpl_->hidden_layer_indices;
      if (std::find(hl.begin(), hl.end(), layer) != hl.end()) {
        captured_hidden_edges[layer] = current;
      }
    }
  }

  // Final RMSNorm
  graph_edge_ptr final_norm_out =
      gb.make_rmsnorm(current, current_name, output_norm_weight_edge, "output_norm.weight",
                      "final_norm_out", "final_norm");

  // Output projection
  graph_edge_ptr logits_edge = gb.make_matmul(final_norm_out, "final_norm_out", output_weight_edge,
                                              "output.weight", "logits", "output_projection");

  auto sync_node = std::make_shared<result_message_sync_node>();

  std::vector<std::string> sync_ids;
  sync_ids.push_back("logits");
  for (const auto& [layer_idx, _] : captured_hidden_edges) {
    sync_ids.push_back("blk." + std::to_string(layer_idx) + ".ffn_residual_out");
  }
  sync_node->set_initial_ids(sync_ids);

  sync_node->set_input(logits_edge, "logits");
  for (const auto& [layer_idx, edge] : captured_hidden_edges) {
    const std::string field = "blk." + std::to_string(layer_idx) + ".ffn_residual_out";
    sync_node->set_input(edge, field);
  }

  pimpl_->graph->add_node(sync_node);
  pimpl_->output_node->set_input(sync_node->get_output(), "default");

  // Add I/O nodes to graph
  pimpl_->graph->add_node(pimpl_->input_node);
  pimpl_->graph->add_node(pimpl_->output_node);

  // Deploy graph
  pimpl_->proc = std::make_unique<graph_proc>();
  pimpl_->proc->deploy(pimpl_->graph);

  // Extract attention nodes from graph
  for (uint32_t i = 0; i < pimpl_->graph->get_node_count(); ++i) {
    if (auto attn = std::dynamic_pointer_cast<grouped_attention_node>(pimpl_->graph->get_node(i))) {
      pimpl_->attention_nodes.push_back(attn);
    }
  }

  // Initialize KV caches after graph is built
  initialize_kv_caches();
}

void llm_engine::wire_io_nodes(graph_edge_ptr input_placeholder, graph_edge_ptr logits_output) {
  pimpl_->input_node = std::make_shared<model_source_node>();
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

  pimpl_->output_node->set_input(logits_output, "logits");

  pimpl_->graph->add_node(pimpl_->input_node);
  pimpl_->graph->add_node(pimpl_->output_node);

  pimpl_->proc = std::make_unique<graph_proc>();
  pimpl_->proc->deploy(pimpl_->graph);
}

void llm_engine::run_inference_step(const std::vector<uint32_t>& tokens) {
  std::promise<std::unordered_map<std::string, dynamic_tensor>> output_promise;
  auto output_future = output_promise.get_future();

  pimpl_->output_node->set_callback(
      [&output_promise](const std::unordered_map<std::string, dynamic_tensor>& result) {
        spdlog::debug("Output callback invoked");
        output_promise.set_value(result);
      });

  // Build input tensor [1, seq_len]
  dynamic_tensor input_tensor(dtype::int32, {1, static_cast<int64_t>(tokens.size())});
  {
    int32_t* data = input_tensor.data_ptr<int32_t>();
    for (size_t j = 0; j < tokens.size(); ++j) {
      data[j] = static_cast<int32_t>(tokens[j]);
    }
  }

  // Build position_ids tensor: [cached_position_, cached_position_+1, ...]
  dynamic_tensor position_ids_tensor(dtype::int64, {static_cast<int64_t>(tokens.size())});
  {
    int64_t* pos_data = position_ids_tensor.data_ptr<int64_t>();
    for (size_t j = 0; j < tokens.size(); ++j) {
      pos_data[j] = pimpl_->cached_position_ + static_cast<int64_t>(j);
    }
  }
  pimpl_->cached_position_ += static_cast<int64_t>(tokens.size());

  pimpl_->input_node->set_tensor("input_ids", input_tensor);
  pimpl_->input_node->set_tensor("position_ids", position_ids_tensor);
  pimpl_->input_node->set_frame_number(++pimpl_->step_counter_);
  pimpl_->input_node->push();

  // Wait for output synchronously
  auto outputs = output_future.get();

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

  // Return logits for the last token position
  const int64_t seq_len = logits.dim(1);
  const int64_t vocab_size = logits.dim(2);
  const float* last_logits = logits.data_ptr<float>() + (seq_len - 1) * vocab_size;
  pimpl_->current_logits_.assign(last_logits, last_logits + vocab_size);

  // Save hidden layer states for speculative decoding
  pimpl_->current_hidden_layers_.clear();
  for (int layer_idx : pimpl_->hidden_layer_indices) {
    const std::string key = "blk." + std::to_string(layer_idx) + ".ffn_residual_out";
    auto hit = outputs.find(key);
    if (hit == outputs.end()) continue;
    const auto& hs = hit->second;
    if (hs.get_dtype() != dtype::float32 || hs.ndim() != 3) continue;
    const int64_t hs_seq_len = hs.dim(1);
    const int64_t hs_hidden = hs.dim(2);
    const float* last_hs = hs.data_ptr<float>() + (hs_seq_len - 1) * hs_hidden;
    pimpl_->current_hidden_layers_[layer_idx].assign(last_hs, last_hs + hs_hidden);
  }
}

void llm_engine::start(const std::vector<uint32_t>& prompt_tokens) {
  if (!pimpl_->loaded) {
    throw std::runtime_error("Model not loaded");
  }
  if (prompt_tokens.empty()) {
    throw std::runtime_error("prompt_tokens must not be empty");
  }

  // Reset state
  reset_kv_caches();
  pimpl_->cached_position_ = 0;
  pimpl_->step_counter_ = 0;

  pimpl_->proc->run();

  run_inference_step(prompt_tokens);
}

void llm_engine::next(uint32_t token) {
  if (!pimpl_->loaded) {
    throw std::runtime_error("Model not loaded");
  }
  run_inference_step({token});
}

void llm_engine::stop() { pimpl_->proc->stop(); }

bool llm_engine::is_loaded() const { return pimpl_->loaded; }

int64_t llm_engine::get_num_layers() const { return pimpl_->num_layers; }

int64_t llm_engine::get_hidden_dim() const { return pimpl_->hidden_dim; }

const std::vector<float>& llm_engine::get_logits() const { return pimpl_->current_logits_; }

const std::vector<float>& llm_engine::get_hidden_layer(int layer_index) const {
  auto it = pimpl_->current_hidden_layers_.find(layer_index);
  if (it == pimpl_->current_hidden_layers_.end()) {
    throw std::runtime_error(
        "Hidden layer " + std::to_string(layer_index) +
        " was not captured. Add it to config::hidden_layer_indices before loading.");
  }
  return it->second;
}

void llm_engine::initialize_kv_caches() {
  if (pimpl_->attention_nodes.empty()) {
    return;
  }

  // Cache shape: [batch=1, num_kv_heads, max_seq_len, head_dim]
  int64_t batch = 1;
  int64_t max_seq_len = std::min(pimpl_->kv_cache_size, pimpl_->max_seq_len);

  size_t cache_size_mb = 0;

  for (auto& attn_node : pimpl_->attention_nodes) {
    // Allocate K and V caches
    dynamic_tensor k_cache(dtype::float32,
                           {batch, pimpl_->num_kv_heads, max_seq_len, pimpl_->head_dim});
    dynamic_tensor v_cache(dtype::float32,
                           {batch, pimpl_->num_kv_heads, max_seq_len, pimpl_->head_dim});

    // Initialize to zero
    std::memset(k_cache.data_ptr<float>(), 0, k_cache.bytes());
    std::memset(v_cache.data_ptr<float>(), 0, v_cache.bytes());

    cache_size_mb += k_cache.bytes() + v_cache.bytes();

    attn_node->set_k_cache(k_cache);
    attn_node->set_v_cache(v_cache);
  }
}

void llm_engine::reset_kv_caches() {
  for (auto& attn_node : pimpl_->attention_nodes) {
    attn_node->reset_cache();
  }
}

}  // namespace coalsack
