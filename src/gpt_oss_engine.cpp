#include "gpt_oss_engine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

#include "gguf_dequant.h"
#include "gguf_loader.h"
#include "gpt2_tokenizer.h"
#include "graph_proc.h"
#include "model_io_nodes.h"
#include "nn_nodes.h"
#include "nn_ops/constant_node.h"
#include "nn_ops/matmul_mixed_node.h"
#include "nn_ops/reshape_node.h"
#include "result_message_nodes.h"

namespace coalsack {

struct gpt_oss_engine::impl {
  // Components
  std::unique_ptr<gguf_loader> loader;
  std::unique_ptr<gpt2_tokenizer> tokenizer;
  std::shared_ptr<subgraph> graph;
  std::unique_ptr<graph_proc> proc;

  // I/O nodes
  std::shared_ptr<model_input_node> input_node;
  std::shared_ptr<model_output_node> output_node;

  // Constant nodes (for weight tensors, need frame source connection)
  std::vector<std::shared_ptr<constant_node>> constant_nodes;

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

  // RoPE config
  float rope_freq_base = 150000.0f;
  float rope_scaling_factor = 32.0f;

  // Weights (loaded from GGUF)
  std::unordered_map<std::string, dynamic_tensor> weights;

  bool loaded = false;
  std::string model_path;
};

gpt_oss_engine::gpt_oss_engine() : pimpl_(std::make_unique<impl>()) {
  pimpl_->loader = std::make_unique<gguf_loader>();
  pimpl_->tokenizer = std::make_unique<gpt2_tokenizer>();
}

gpt_oss_engine::~gpt_oss_engine() = default;

bool gpt_oss_engine::load(const std::string& gguf_path) {
  if (pimpl_->loaded) {
    std::cerr << "Error: Model already loaded\n";
    return false;
  }

  pimpl_->model_path = gguf_path;

  // Load GGUF file
  std::cout << "Loading GGUF file: " << gguf_path << "\n";
  if (!pimpl_->loader->load(gguf_path)) {
    std::cerr << "Error: Failed to load GGUF file\n";
    return false;
  }

  // Load tokenizer from GGUF
  std::cout << "Loading tokenizer...\n";
  if (!pimpl_->tokenizer->load_from_gguf(*pimpl_->loader)) {
    std::cerr << "Error: Failed to load tokenizer\n";
    return false;
  }

  // Load model config from GGUF metadata
  load_config_from_gguf();

  // Load weights from GGUF
  std::cout << "Loading weights...\n";
  if (!load_weights_from_gguf()) {
    std::cerr << "Error: Failed to load weights\n";
    return false;
  }

  // Build inference graph
  std::cout << "Building transformer graph...\n";
  build_transformer_graph();

  pimpl_->loaded = true;
  std::cout << "Model loaded successfully\n";
  std::cout << "  Vocab size: " << pimpl_->vocab_size << "\n";
  std::cout << "  Layers: " << pimpl_->num_layers << "\n";
  std::cout << "  Hidden dim: " << pimpl_->hidden_dim << "\n";
  std::cout << "  Q heads: " << pimpl_->num_q_heads << "\n";
  std::cout << "  KV heads: " << pimpl_->num_kv_heads << "\n";
  std::cout << "  Experts: " << pimpl_->num_experts << " (top-" << pimpl_->expert_top_k << ")\n";

  return true;
}

void gpt_oss_engine::load_config_from_gguf() {
  // Read model config from GGUF metadata using typed getters
  auto& loader = *pimpl_->loader;

  // Read config from GGUF (llama format keys)
  if (auto val = loader.get_uint32("llama.block_count")) {
    pimpl_->num_layers = *val;
  }
  if (auto val = loader.get_uint32("llama.embedding_length")) {
    pimpl_->hidden_dim = *val;
  }
  if (auto val = loader.get_uint32("llama.attention.head_count")) {
    pimpl_->num_q_heads = *val;
  }
  if (auto val = loader.get_uint32("llama.attention.head_count_kv")) {
    pimpl_->num_kv_heads = *val;
  }
  if (auto val = loader.get_uint32("llama.vocab_size")) {
    pimpl_->vocab_size = *val;
  }
  if (auto val = loader.get_uint32("llama.context_length")) {
    pimpl_->max_seq_len = *val;
  }
  if (auto val = loader.get_uint32("llama.expert_count")) {
    pimpl_->num_experts = *val;
  }
  if (auto val = loader.get_uint32("llama.expert_used_count")) {
    pimpl_->expert_top_k = *val;
  }
  if (auto val = loader.get_uint32("llama.expert_feed_forward_length")) {
    pimpl_->expert_ffn_dim = *val;
  }

  // Try to read head_dim from GGUF metadata
  if (auto val = loader.get_uint32("llama.attention.head_dim")) {
    pimpl_->head_dim = *val;
  } else if (pimpl_->num_q_heads > 0) {
    // Fallback: compute from hidden_dim / num_q_heads
    pimpl_->head_dim = pimpl_->hidden_dim / pimpl_->num_q_heads;
  }

  // RoPE config
  if (auto val = loader.get_float32("llama.rope.freq_base")) {
    pimpl_->rope_freq_base = *val;
  }
  if (auto val = loader.get_float32("llama.rope.scaling.factor")) {
    pimpl_->rope_scaling_factor = *val;
  }

  std::cout << "  Config loaded from GGUF metadata\n";
}

bool gpt_oss_engine::load_weights_from_gguf() {
  auto& loader = *pimpl_->loader;
  const auto& tensor_names = loader.get_tensor_names();
  const std::string& file_path = loader.get_file_path();

  std::cout << "  Total tensors in GGUF: " << tensor_names.size() << "\n";

  // Open file for reading tensor data
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    std::cerr << "ERROR: Failed to open file for tensor loading: " << file_path << "\n";
    return false;
  }

  size_t loaded_count = 0;
  size_t dequantized_count = 0;
  size_t skipped_count = 0;

  size_t tensor_idx = 0;
  try {
    for (const auto& name : tensor_names) {
      ++tensor_idx;
      auto info_opt = loader.get_tensor_info(name);
      if (!info_opt) {
        std::cerr << "WARNING: Failed to get tensor info for: " << name << "\n";
        continue;
      }
      const auto& info = *info_opt;

      // Build shape from tensor info
      std::vector<int64_t> shape;
      for (auto dim : info.shape) {
        shape.push_back(static_cast<int64_t>(dim));
      }

      // Calculate number of elements
      int64_t numel = 1;
      for (auto dim : shape) {
        numel *= dim;
      }

      // Read raw data from file
      file.seekg(info.offset);
      if (!file) {
        std::cerr << "ERROR: Failed to seek to tensor data for: " << name << "\n";
        continue;
      }

      std::vector<uint8_t> raw_data(info.size);
      file.read(reinterpret_cast<char*>(raw_data.data()), info.size);
      if (!file) {
        std::cerr << "ERROR: Failed to read tensor data for: " << name << "\n";
        continue;
      }

      // Determine output dtype based on GGUF tensor type
      dtype output_dtype = dtype::float32;  // Default

      if (info.type == ggml_type::F16 || info.type == ggml_type::MXFP4) {
        output_dtype = dtype::float16;  // Keep as float16 for memory savings
      }

      dynamic_tensor tensor(output_dtype, shape);

      if (info.type == ggml_type::F32) {
        // Direct copy for float32
        float* data = tensor.data_ptr<float>();
        std::memcpy(data, raw_data.data(), info.size);
        ++loaded_count;
      } else if (info.type == ggml_type::F16) {
        // Direct copy for float16 (no conversion)
        uint16_t* data = tensor.data_ptr<uint16_t>();
        std::memcpy(data, raw_data.data(), info.size);
        ++loaded_count;
      } else if (info.type == ggml_type::MXFP4) {
        // Convert MXFP4 → float16
        uint16_t* data = tensor.data_ptr<uint16_t>();
        if (dequantize_mxfp4_to_fp16(raw_data.data(), data, numel)) {
          ++dequantized_count;
        } else {
          std::cerr << "WARNING: Failed to convert MXFP4 to float16 for: " << name << "\n";
          ++skipped_count;
          continue;
        }
      } else {
        // Other quantized formats → dequantize to float32
        float* data = tensor.data_ptr<float>();
        if (dequantize_tensor(raw_data.data(), data, numel, info.type)) {
          ++dequantized_count;
        } else {
          std::cerr << "WARNING: Unsupported quantization type for: " << name << "\n";
          ++skipped_count;
          continue;
        }
      }

      pimpl_->weights[name] = std::move(tensor);
    }
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION during weight loading at tensor " << tensor_idx << ": " << e.what()
              << "\n";
    return false;
  }

  std::cout << "  Loaded " << loaded_count << " float32 tensors\n";
  std::cout << "  Dequantized " << dequantized_count << " quantized tensors\n";
  if (skipped_count > 0) {
    std::cout << "  Skipped " << skipped_count << " unsupported tensors\n";
  }

  // Check for required weights
  std::vector<std::string> required_weights = {"token_embd.weight", "output.weight"};

  for (const auto& req : required_weights) {
    if (pimpl_->weights.find(req) == pimpl_->weights.end()) {
      std::cerr << "ERROR: Required weight '" << req << "' not found\n";
      return false;
    }
  }

  return true;
}

void gpt_oss_engine::build_transformer_graph() {
  // Full 24-layer transformer: embedding → 24 layers → final norm → output

  pimpl_->graph = std::make_shared<subgraph>();

  // Create I/O nodes first
  pimpl_->input_node = std::make_shared<model_input_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  // Collect required weights for all 24 layers
  std::vector<std::string> required_weights = {
      "token_embd.weight",
      "output_norm.weight",
      "output.weight",
  };

  // Add weights for all 24 layers
  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_prefix = "blk." + std::to_string(layer);
    required_weights.push_back(layer_prefix + ".attn_norm.weight");
    required_weights.push_back(layer_prefix + ".attn_q.weight");
    required_weights.push_back(layer_prefix + ".attn_k.weight");
    required_weights.push_back(layer_prefix + ".attn_v.weight");
    required_weights.push_back(layer_prefix + ".attn_output.weight");
    required_weights.push_back(layer_prefix + ".post_attention_norm.weight");
    required_weights.push_back(layer_prefix + ".ffn_gate_inp.weight");
    required_weights.push_back(layer_prefix + ".ffn_up_exps.weight");
    required_weights.push_back(layer_prefix + ".ffn_gate_exps.weight");
    required_weights.push_back(layer_prefix + ".ffn_down_exps.weight");
  }

  // Add all required weights to input_node
  for (const auto& name : required_weights) {
    auto it = pimpl_->weights.find(name);
    if (it == pimpl_->weights.end()) {
      std::cerr << "WARNING: Weight not found (continuing anyway): " << name << "\n";
      continue;
    }
    pimpl_->input_node->set_tensor(name, it->second);
  }

  // Create extractor to extract fields from input_node
  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  // Extract input_ids and common weights
  auto input_ids_edge = extractor->add_output("input_ids");
  auto token_embd_weight_edge_raw = extractor->add_output("token_embd.weight");
  auto output_norm_weight_edge = extractor->add_output("output_norm.weight");
  auto output_weight_edge = extractor->add_output("output.weight");

  // Transpose token_embd.weight from [hidden_dim, vocab_size] to [vocab_size, hidden_dim]
  auto embd_transpose = std::make_shared<transpose_node>();
  embd_transpose->set_perm({1, 0});
  embd_transpose->set_input(token_embd_weight_edge_raw, "default");
  embd_transpose->set_input_name("token_embd.weight");
  embd_transpose->set_output_name("token_embd.weight_t");
  embd_transpose->set_node_name("embd_transpose");
  pimpl_->graph->add_node(embd_transpose);
  auto token_embd_weight_edge = embd_transpose->get_output("default");

  // Extract all layer weights
  std::vector<std::unordered_map<std::string, graph_edge_ptr>> layer_weights(pimpl_->num_layers);
  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_prefix = "blk." + std::to_string(layer);
    std::vector<std::string> layer_weight_names = {
        layer_prefix + ".attn_norm.weight",
        layer_prefix + ".attn_q.weight",
        layer_prefix + ".attn_k.weight",
        layer_prefix + ".attn_v.weight",
        layer_prefix + ".attn_output.weight",
        layer_prefix + ".post_attention_norm.weight",
        layer_prefix + ".ffn_gate_inp.weight",
        layer_prefix + ".ffn_up_exps.weight",
        layer_prefix + ".ffn_gate_exps.weight",
        layer_prefix + ".ffn_down_exps.weight",
    };

    for (const auto& name : layer_weight_names) {
      if (pimpl_->weights.find(name) != pimpl_->weights.end()) {
        layer_weights[layer][name] = extractor->add_output(name);
      }
    }
  }

  // ========== 1. Embedding layer ==========
  auto embedding = std::make_shared<embedding_lookup_node>();
  auto emb_sync = std::make_shared<result_message_sync_node>();
  emb_sync->set_input(input_ids_edge, "input_ids");
  emb_sync->set_input(token_embd_weight_edge, "token_embd.weight_t");
  emb_sync->set_initial_ids({"input_ids", "token_embd.weight_t"});
  pimpl_->graph->add_node(emb_sync);

  embedding->set_input(emb_sync->get_output(), "default");
  embedding->set_input_names("input_ids", "token_embd.weight_t");
  embedding->set_output_name("embeddings");
  embedding->set_node_name("embedding_lookup");
  pimpl_->graph->add_node(embedding);

  graph_edge_ptr current = embedding->get_output("default");
  std::string current_name = "embeddings";

  // ========== 2. All 24 Transformer Layers ==========
  for (int layer = 0; layer < pimpl_->num_layers; ++layer) {
    std::string layer_str = std::to_string(layer);
    std::string layer_prefix = "blk." + layer_str;

    // Calculate head_dim for this layer from attn_q.weight shape
    int64_t layer_head_dim = pimpl_->head_dim;  // Default fallback
    auto q_weight_it = pimpl_->weights.find(layer_prefix + ".attn_q.weight");
    if (q_weight_it != pimpl_->weights.end() && pimpl_->num_q_heads > 0) {
      const auto& q_shape = q_weight_it->second.shape();
      if (q_shape.size() == 2 && q_shape[0] == pimpl_->hidden_dim) {
        int64_t q_output_dim = q_shape[1];
        layer_head_dim = q_output_dim / pimpl_->num_q_heads;
      }
    }

    // Store input to this layer for residual connections
    graph_edge_ptr layer_input = current;
    std::string layer_input_name = current_name;

    // 2.1 Input RMSNorm (before attention)
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

    // 2.2 Q/K/V projections
    // Q projection
    auto q_proj = std::make_shared<matmul_mixed_node>();
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
    graph_edge_ptr q_out = q_proj->get_output("default");

    // K projection
    auto k_proj = std::make_shared<matmul_mixed_node>();
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
    graph_edge_ptr k_out = k_proj->get_output("default");

    // V projection
    auto v_proj = std::make_shared<matmul_mixed_node>();
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
    graph_edge_ptr v_out = v_proj->get_output("default");

    // 2.3 Reshape Q/K for RoPE (3D → 4D → transpose → RoPE → transpose → 4D → 3D)

    // Create shape constants for reshape operations
    // Q: [batch, seq_len, 2880] → [batch, seq_len, 64, 45]
    dynamic_tensor q_reshape_shape(dtype::int64, {4});
    auto q_reshape_shape_data = q_reshape_shape.data_ptr<int64_t>();
    q_reshape_shape_data[0] = 0;                      // Copy batch from input[0]
    q_reshape_shape_data[1] = 0;                      // Copy seq_len from input[1]
    q_reshape_shape_data[2] = pimpl_->num_q_heads;   // 64 (explicit)
    q_reshape_shape_data[3] = -1;                     // 45 (inferred from total elements)

    auto q_reshape_shape_const = std::make_shared<constant_node>(q_reshape_shape, layer_prefix + ".q_reshape_shape");
    q_reshape_shape_const->set_input(q_out, "default");
    pimpl_->graph->add_node(q_reshape_shape_const);
    auto q_reshape_shape_edge = q_reshape_shape_const->get_output("default");

    // K: [batch, seq_len, 360] → [batch, seq_len, 8, 45]
    dynamic_tensor k_reshape_shape(dtype::int64, {4});
    auto k_reshape_shape_data = k_reshape_shape.data_ptr<int64_t>();
    k_reshape_shape_data[0] = 0;                      // Copy batch from input[0]
    k_reshape_shape_data[1] = 0;                      // Copy seq_len from input[1]
    k_reshape_shape_data[2] = pimpl_->num_kv_heads;  // 8 (explicit)
    k_reshape_shape_data[3] = -1;                     // 45 (inferred from total elements)

    auto k_reshape_shape_const = std::make_shared<constant_node>(k_reshape_shape, layer_prefix + ".k_reshape_shape");
    k_reshape_shape_const->set_input(k_out, "default");
    pimpl_->graph->add_node(k_reshape_shape_const);
    auto k_reshape_shape_edge = k_reshape_shape_const->get_output("default");

    // Reshape back: [batch, seq_len, num_heads, head_dim] → [batch, seq_len, -1]
    dynamic_tensor reshape_back_shape(dtype::int64, {3});
    auto reshape_back_shape_data = reshape_back_shape.data_ptr<int64_t>();
    reshape_back_shape_data[0] = 0;   // Copy batch
    reshape_back_shape_data[1] = 0;   // Copy seq_len
    reshape_back_shape_data[2] = -1;  // Infer from total elements

    auto reshape_back_shape_const = std::make_shared<constant_node>(reshape_back_shape, layer_prefix + ".reshape_back_shape");
    reshape_back_shape_const->set_input(q_out, "default");
    pimpl_->graph->add_node(reshape_back_shape_const);
    auto reshape_back_shape_edge = reshape_back_shape_const->get_output("default");

    // Process Q: reshape → transpose → RoPE → transpose → reshape
    auto q_reshape_4d = std::make_shared<reshape_node>();
    auto q_reshape_4d_sync = std::make_shared<result_message_sync_node>();
    q_reshape_4d_sync->set_input(q_out, layer_prefix + ".q_proj_out");
    q_reshape_4d_sync->set_input(q_reshape_shape_edge, layer_prefix + ".q_reshape_shape");
    q_reshape_4d_sync->set_initial_ids({layer_prefix + ".q_proj_out", layer_prefix + ".q_reshape_shape"});
    pimpl_->graph->add_node(q_reshape_4d_sync);

    q_reshape_4d->set_input(q_reshape_4d_sync->get_output(), "default");
    q_reshape_4d->set_input_names(layer_prefix + ".q_proj_out", layer_prefix + ".q_reshape_shape");
    q_reshape_4d->set_output_name(layer_prefix + ".q_4d");
    q_reshape_4d->set_node_name(layer_prefix + ".q_reshape_4d");
    pimpl_->graph->add_node(q_reshape_4d);
    graph_edge_ptr q_4d = q_reshape_4d->get_output("default");

    // Transpose Q: [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
    auto q_transpose = std::make_shared<transpose_node>();
    q_transpose->set_perm({0, 2, 1, 3});
    q_transpose->set_input(q_4d, "default");
    q_transpose->set_input_name(layer_prefix + ".q_4d");
    q_transpose->set_output_name(layer_prefix + ".q_transposed");
    q_transpose->set_node_name(layer_prefix + ".q_transpose");
    pimpl_->graph->add_node(q_transpose);
    graph_edge_ptr q_transposed = q_transpose->get_output("default");

    // RoPE Q
    auto rope_q = std::make_shared<rope_node>();
    rope_q->set_config(layer_head_dim, pimpl_->max_seq_len, pimpl_->rope_freq_base,
                       pimpl_->rope_scaling_factor);
    rope_q->set_input(q_transposed, "default");
    rope_q->set_input_name(layer_prefix + ".q_transposed");
    rope_q->set_output_name(layer_prefix + ".q_rope");
    rope_q->set_node_name(layer_prefix + ".rope_q");
    pimpl_->graph->add_node(rope_q);
    graph_edge_ptr q_rope = rope_q->get_output("default");

    // Transpose Q back: [batch, num_heads, seq_len, head_dim] → [batch, seq_len, num_heads, head_dim]
    auto q_transpose_back = std::make_shared<transpose_node>();
    q_transpose_back->set_perm({0, 2, 1, 3});
    q_transpose_back->set_input(q_rope, "default");
    q_transpose_back->set_input_name(layer_prefix + ".q_rope");
    q_transpose_back->set_output_name(layer_prefix + ".q_rope_4d");
    q_transpose_back->set_node_name(layer_prefix + ".q_transpose_back");
    pimpl_->graph->add_node(q_transpose_back);
    graph_edge_ptr q_rope_4d = q_transpose_back->get_output("default");

    // Reshape Q back to 3D
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

    // Process K: reshape → transpose → RoPE → transpose → reshape
    auto k_reshape_4d = std::make_shared<reshape_node>();
    auto k_reshape_4d_sync = std::make_shared<result_message_sync_node>();
    k_reshape_4d_sync->set_input(k_out, layer_prefix + ".k_proj_out");
    k_reshape_4d_sync->set_input(k_reshape_shape_edge, layer_prefix + ".k_reshape_shape");
    k_reshape_4d_sync->set_initial_ids({layer_prefix + ".k_proj_out", layer_prefix + ".k_reshape_shape"});
    pimpl_->graph->add_node(k_reshape_4d_sync);

    k_reshape_4d->set_input(k_reshape_4d_sync->get_output(), "default");
    k_reshape_4d->set_input_names(layer_prefix + ".k_proj_out", layer_prefix + ".k_reshape_shape");
    k_reshape_4d->set_output_name(layer_prefix + ".k_4d");
    k_reshape_4d->set_node_name(layer_prefix + ".k_reshape_4d");
    pimpl_->graph->add_node(k_reshape_4d);
    graph_edge_ptr k_4d = k_reshape_4d->get_output("default");

    // Transpose K: [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
    auto k_transpose = std::make_shared<transpose_node>();
    k_transpose->set_perm({0, 2, 1, 3});
    k_transpose->set_input(k_4d, "default");
    k_transpose->set_input_name(layer_prefix + ".k_4d");
    k_transpose->set_output_name(layer_prefix + ".k_transposed");
    k_transpose->set_node_name(layer_prefix + ".k_transpose");
    pimpl_->graph->add_node(k_transpose);
    graph_edge_ptr k_transposed = k_transpose->get_output("default");

    // RoPE K
    auto rope_k = std::make_shared<rope_node>();
    rope_k->set_config(layer_head_dim, pimpl_->max_seq_len, pimpl_->rope_freq_base,
                       pimpl_->rope_scaling_factor);
    rope_k->set_input(k_transposed, "default");
    rope_k->set_input_name(layer_prefix + ".k_transposed");
    rope_k->set_output_name(layer_prefix + ".k_rope");
    rope_k->set_node_name(layer_prefix + ".rope_k");
    pimpl_->graph->add_node(rope_k);
    graph_edge_ptr k_rope = rope_k->get_output("default");

    // Transpose K back: [batch, num_heads, seq_len, head_dim] → [batch, seq_len, num_heads, head_dim]
    auto k_transpose_back = std::make_shared<transpose_node>();
    k_transpose_back->set_perm({0, 2, 1, 3});
    k_transpose_back->set_input(k_rope, "default");
    k_transpose_back->set_input_name(layer_prefix + ".k_rope");
    k_transpose_back->set_output_name(layer_prefix + ".k_rope_4d");
    k_transpose_back->set_node_name(layer_prefix + ".k_transpose_back");
    pimpl_->graph->add_node(k_transpose_back);
    graph_edge_ptr k_rope_4d = k_transpose_back->get_output("default");

    // Reshape K back to 3D
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

    // 2.4 Grouped Query Attention
    auto attn = std::make_shared<grouped_attention_node>();
    attn->set_config(pimpl_->num_q_heads, pimpl_->num_kv_heads, layer_head_dim);

    auto attn_sync = std::make_shared<result_message_sync_node>();
    attn_sync->set_input(q_rope_out, layer_prefix + ".q_rope_out");
    attn_sync->set_input(k_rope_out, layer_prefix + ".k_rope_out");
    attn_sync->set_input(v_out, layer_prefix + ".v_proj_out");
    attn_sync->set_initial_ids({layer_prefix + ".q_rope_out", layer_prefix + ".k_rope_out", layer_prefix + ".v_proj_out"});
    pimpl_->graph->add_node(attn_sync);

    attn->set_input(attn_sync->get_output(), "default");
    attn->set_input_names({layer_prefix + ".q_rope_out", layer_prefix + ".k_rope_out", layer_prefix + ".v_proj_out"});
    attn->set_output_name(layer_prefix + ".attn_out");
    attn->set_node_name(layer_prefix + ".grouped_attn");
    pimpl_->graph->add_node(attn);
    graph_edge_ptr attn_out = attn->get_output("default");

    // 2.5 Attention output projection
    auto attn_proj = std::make_shared<matmul_mixed_node>();
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
    graph_edge_ptr attn_proj_out = attn_proj->get_output("default");

    // 2.6 Attention residual connection
    auto attn_residual = std::make_shared<add_node>();
    auto attn_res_sync = std::make_shared<result_message_sync_node>();
    attn_res_sync->set_input(layer_input, layer_input_name);
    attn_res_sync->set_input(attn_proj_out, layer_prefix + ".attn_proj_out");
    attn_res_sync->set_initial_ids({layer_input_name, layer_prefix + ".attn_proj_out"});
    pimpl_->graph->add_node(attn_res_sync);

    attn_residual->set_input(attn_res_sync->get_output(), "default");
    attn_residual->set_input_names(layer_input_name, layer_prefix + ".attn_proj_out");
    attn_residual->set_output_name(layer_prefix + ".attn_residual_out");
    attn_residual->set_node_name(layer_prefix + ".attn_residual");
    pimpl_->graph->add_node(attn_residual);
    graph_edge_ptr attn_res_out = attn_residual->get_output("default");

    // 2.7 FFN RMSNorm (before MoE)
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

    // 2.8 MoE Router (top-k expert selection)
    auto router = std::make_shared<moe_router_node>();
    router->set_config(pimpl_->num_experts, pimpl_->expert_top_k);

    auto router_sync = std::make_shared<result_message_sync_node>();
    router_sync->set_input(ffn_norm_out, layer_prefix + ".ffn_norm_out");
    router_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_gate_inp.weight"], layer_prefix + ".ffn_gate_inp.weight");
    router_sync->set_initial_ids({layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_gate_inp.weight"});
    pimpl_->graph->add_node(router_sync);

    router->set_input(router_sync->get_output(), "default");
    router->set_input_names(layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_gate_inp.weight");
    router->set_output_name(layer_prefix + ".router_out");
    router->set_node_name(layer_prefix + ".moe_router");
    pimpl_->graph->add_node(router);
    graph_edge_ptr router_out = router->get_output("default");

    // 2.9 Expert MLPs (32 experts in parallel)
    std::vector<graph_edge_ptr> expert_outputs;
    for (int expert_id = 0; expert_id < pimpl_->num_experts; ++expert_id) {
      auto expert = std::make_shared<expert_mlp_node>(expert_id);

      auto expert_sync = std::make_shared<result_message_sync_node>();
      expert_sync->set_input(ffn_norm_out, layer_prefix + ".ffn_norm_out");
      expert_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_up_exps.weight"], layer_prefix + ".ffn_up_exps.weight");
      expert_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_gate_exps.weight"], layer_prefix + ".ffn_gate_exps.weight");
      expert_sync->set_input(layer_weights[layer][layer_prefix + ".ffn_down_exps.weight"], layer_prefix + ".ffn_down_exps.weight");
      expert_sync->set_initial_ids({layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_up_exps.weight", layer_prefix + ".ffn_gate_exps.weight", layer_prefix + ".ffn_down_exps.weight"});
      pimpl_->graph->add_node(expert_sync);

      expert->set_input(expert_sync->get_output(), "default");
      expert->set_input_names({layer_prefix + ".ffn_norm_out", layer_prefix + ".ffn_up_exps.weight", layer_prefix + ".ffn_gate_exps.weight", layer_prefix + ".ffn_down_exps.weight"});
      expert->set_output_name(layer_prefix + ".expert_" + std::to_string(expert_id) + "_out");
      expert->set_node_name(layer_prefix + ".expert_" + std::to_string(expert_id));
      pimpl_->graph->add_node(expert);
      expert_outputs.push_back(expert->get_output("default"));
    }

    // 2.10 Expert merge (weighted sum of top-k experts)
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

    // 2.11 FFN residual connection
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

  // ========== 3. Final RMSNorm ==========
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

  // ========== 4. Output projection ==========
  auto output_proj = std::make_shared<matmul_mixed_node>();

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

  // ========== 5. Output sync and I/O nodes ==========
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

  std::cout << "✓ Full 24-layer transformer graph built\n";
}

void gpt_oss_engine::build_minimal_test_graph() {
  // Minimal graph: embedding → output projection
  // No transformer layers (for testing graph infrastructure)

  pimpl_->graph = std::make_shared<subgraph>();

  // Create I/O nodes first
  pimpl_->input_node = std::make_shared<model_input_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  // Add weights to input_node (like test_superpoint_lightglue does)
  std::vector<std::string> required_weights = {"token_embd.weight", "output.weight"};
  for (const auto& name : required_weights) {
    auto it = pimpl_->weights.find(name);
    if (it == pimpl_->weights.end()) {
      throw std::runtime_error("Required weight not found: " + name);
    }
    pimpl_->input_node->set_tensor(name, it->second);
  }

  // Create extractor to extract fields from input_node
  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  // Add outputs for each field
  graph_edge_ptr input_ids_edge = extractor->add_output("input_ids");
  graph_edge_ptr token_embd_weight_edge_raw = extractor->add_output("token_embd.weight");
  graph_edge_ptr output_weight_edge = extractor->add_output("output.weight");

  // Transpose token_embd.weight from [hidden_dim, vocab_size] to [vocab_size, hidden_dim]
  auto embd_transpose = std::make_shared<transpose_node>();
  embd_transpose->set_perm({1, 0});  // Swap dimensions
  embd_transpose->set_input(token_embd_weight_edge_raw, "default");
  embd_transpose->set_input_name("token_embd.weight");
  embd_transpose->set_output_name("token_embd.weight_t");
  embd_transpose->set_node_name("embd_transpose");
  pimpl_->graph->add_node(embd_transpose);

  graph_edge_ptr token_embd_weight_edge = embd_transpose->get_output("default");

  // 1. Embedding layer
  auto embedding = std::make_shared<embedding_lookup_node>();

  auto emb_sync = std::make_shared<result_message_sync_node>();
  emb_sync->set_input(input_ids_edge, "input_ids");
  emb_sync->set_input(token_embd_weight_edge, "token_embd.weight_t");
  emb_sync->set_initial_ids({"input_ids", "token_embd.weight_t"});
  pimpl_->graph->add_node(emb_sync);

  embedding->set_input(emb_sync->get_output(), "default");
  embedding->set_input_names("input_ids", "token_embd.weight_t");
  embedding->set_output_name("embeddings");
  embedding->set_node_name("embedding_lookup");
  pimpl_->graph->add_node(embedding);

  graph_edge_ptr embeddings_edge = embedding->get_output("default");

  // 2. Output projection (matmul: embeddings × output.weight → logits)
  // embeddings: [batch, seq_len, hidden_dim] = [1, 1, 2880]
  // output.weight: [hidden_dim, vocab_size] = [2880, 201088]
  // logits: [batch, seq_len, vocab_size] = [1, 1, 201088]
  auto output_proj = std::make_shared<matmul_mixed_node>();

  auto proj_sync = std::make_shared<result_message_sync_node>();
  proj_sync->set_input(embeddings_edge, "embeddings");
  proj_sync->set_input(output_weight_edge, "output.weight");
  proj_sync->set_initial_ids({"embeddings", "output.weight"});
  pimpl_->graph->add_node(proj_sync);

  output_proj->set_input(proj_sync->get_output(), "default");
  output_proj->set_input_names("embeddings", "output.weight");
  output_proj->set_output_name("logits");
  output_proj->set_node_name("output_projection");
  pimpl_->graph->add_node(output_proj);

  graph_edge_ptr logits_edge = output_proj->get_output("default");

  // Create output sync node
  auto output_sync = std::make_shared<result_message_sync_node>();
  pimpl_->graph->add_node(output_sync);
  output_sync->set_input(logits_edge, "logits");
  output_sync->set_initial_ids({"logits"});

  // Connect to output_node
  pimpl_->output_node->set_input(output_sync->get_output(), "default");

  // Add I/O nodes to graph
  pimpl_->graph->add_node(pimpl_->input_node);
  pimpl_->graph->add_node(pimpl_->output_node);

  // Deploy graph
  pimpl_->proc = std::make_unique<graph_proc>();
  pimpl_->proc->deploy(pimpl_->graph);
}

void gpt_oss_engine::build_single_layer_test_graph() {
  // 1-layer transformer: embedding → Layer 0 → final norm → output
  // This tests all transformer components with a single layer

  pimpl_->graph = std::make_shared<subgraph>();

  // Calculate head_dim for layer 0 from attn_q.weight shape
  int64_t layer0_head_dim = pimpl_->head_dim;  // Default fallback
  auto q_weight_it = pimpl_->weights.find("blk.0.attn_q.weight");
  if (q_weight_it != pimpl_->weights.end() && pimpl_->num_q_heads > 0) {
    const auto& q_shape = q_weight_it->second.shape();
    if (q_shape.size() == 2 && q_shape[0] == pimpl_->hidden_dim) {
      int64_t q_output_dim = q_shape[1];
      layer0_head_dim = q_output_dim / pimpl_->num_q_heads;
      std::cout << "  Layer 0 head_dim: " << layer0_head_dim
                << " (from attn_q.weight shape [" << q_shape[0] << ", " << q_shape[1]
                << "] / " << pimpl_->num_q_heads << " heads)\n";
    }
  }

  // Create I/O nodes first
  pimpl_->input_node = std::make_shared<model_input_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  // Collect required weights for layer 0
  std::vector<std::string> required_weights = {
      "token_embd.weight",
      "output_norm.weight",
      "output.weight",
      // Layer 0 attention weights
      "blk.0.attn_norm.weight",
      "blk.0.attn_q.weight",
      "blk.0.attn_k.weight",
      "blk.0.attn_v.weight",
      "blk.0.attn_output.weight",
      // Layer 0 FFN weights
      "blk.0.post_attention_norm.weight",  // FFN norm
      "blk.0.ffn_gate_inp.weight",         // MoE router gate
      "blk.0.ffn_up_exps.weight",          // All experts up projection
      "blk.0.ffn_gate_exps.weight",        // All experts gate projection
      "blk.0.ffn_down_exps.weight",        // All experts down projection
  };

  // Add all required weights to input_node
  for (const auto& name : required_weights) {
    auto it = pimpl_->weights.find(name);
    if (it == pimpl_->weights.end()) {
      std::cerr << "WARNING: Weight not found (continuing anyway): " << name << "\n";
      continue;  // Continue even if weight missing for now
    }
    pimpl_->input_node->set_tensor(name, it->second);
  }

  // Create extractor to extract fields from input_node
  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  // Extract input_ids and weights
  auto input_ids_edge = extractor->add_output("input_ids");
  auto token_embd_weight_edge_raw = extractor->add_output("token_embd.weight");
  auto output_norm_weight_edge = extractor->add_output("output_norm.weight");
  auto output_weight_edge = extractor->add_output("output.weight");

  // Transpose token_embd.weight from [hidden_dim, vocab_size] to [vocab_size, hidden_dim]
  auto embd_transpose = std::make_shared<transpose_node>();
  embd_transpose->set_perm({1, 0});
  embd_transpose->set_input(token_embd_weight_edge_raw, "default");
  embd_transpose->set_input_name("token_embd.weight");
  embd_transpose->set_output_name("token_embd.weight_t");
  embd_transpose->set_node_name("embd_transpose");
  pimpl_->graph->add_node(embd_transpose);
  auto token_embd_weight_edge = embd_transpose->get_output("default");

  // Extract layer 0 weights
  std::unordered_map<std::string, graph_edge_ptr> layer0_weights;
  std::vector<std::string> layer0_weight_names = {
      "blk.0.attn_norm.weight",
      "blk.0.attn_q.weight",
      "blk.0.attn_k.weight",
      "blk.0.attn_v.weight",
      "blk.0.attn_output.weight",
      "blk.0.post_attention_norm.weight",
      "blk.0.ffn_gate_inp.weight",
      "blk.0.ffn_up_exps.weight",
      "blk.0.ffn_gate_exps.weight",
      "blk.0.ffn_down_exps.weight",
  };

  for (const auto& name : layer0_weight_names) {
    if (pimpl_->weights.find(name) != pimpl_->weights.end()) {
      layer0_weights[name] = extractor->add_output(name);
    }
  }

  // ========== 1. Embedding layer ==========
  auto embedding = std::make_shared<embedding_lookup_node>();
  auto emb_sync = std::make_shared<result_message_sync_node>();
  emb_sync->set_input(input_ids_edge, "input_ids");
  emb_sync->set_input(token_embd_weight_edge, "token_embd.weight_t");
  emb_sync->set_initial_ids({"input_ids", "token_embd.weight_t"});
  pimpl_->graph->add_node(emb_sync);

  embedding->set_input(emb_sync->get_output(), "default");
  embedding->set_input_names("input_ids", "token_embd.weight_t");
  embedding->set_output_name("embeddings");
  embedding->set_node_name("embedding_lookup");
  pimpl_->graph->add_node(embedding);

  graph_edge_ptr current = embedding->get_output("default");

  // ========== 2. Layer 0 Transformer Block ==========

  // 2.1 Input RMSNorm (before attention)
  auto input_norm = std::make_shared<rmsnorm_node>();
  input_norm->set_epsilon(1e-5f);

  auto norm_sync = std::make_shared<result_message_sync_node>();
  norm_sync->set_input(current, "embeddings");
  norm_sync->set_input(layer0_weights["blk.0.attn_norm.weight"], "blk.0.attn_norm.weight");
  norm_sync->set_initial_ids({"embeddings", "blk.0.attn_norm.weight"});
  pimpl_->graph->add_node(norm_sync);

  input_norm->set_input(norm_sync->get_output(), "default");
  input_norm->set_input_names("embeddings", "blk.0.attn_norm.weight");
  input_norm->set_output_name("blk.0.attn_norm_out");
  input_norm->set_node_name("blk.0.attn_norm");
  pimpl_->graph->add_node(input_norm);

  graph_edge_ptr norm_out = input_norm->get_output("default");

  // 2.2 Q/K/V projections
  // Q projection
  auto q_proj = std::make_shared<matmul_mixed_node>();
  auto q_sync = std::make_shared<result_message_sync_node>();
  q_sync->set_input(norm_out, "blk.0.attn_norm_out");
  q_sync->set_input(layer0_weights["blk.0.attn_q.weight"], "blk.0.attn_q.weight");
  q_sync->set_initial_ids({"blk.0.attn_norm_out", "blk.0.attn_q.weight"});
  pimpl_->graph->add_node(q_sync);

  q_proj->set_input(q_sync->get_output(), "default");
  q_proj->set_input_names("blk.0.attn_norm_out", "blk.0.attn_q.weight");
  q_proj->set_output_name("blk.0.q_proj_out");
  q_proj->set_node_name("blk.0.q_proj");
  pimpl_->graph->add_node(q_proj);
  graph_edge_ptr q_out = q_proj->get_output("default");

  // K projection
  auto k_proj = std::make_shared<matmul_mixed_node>();
  auto k_sync = std::make_shared<result_message_sync_node>();
  k_sync->set_input(norm_out, "blk.0.attn_norm_out");
  k_sync->set_input(layer0_weights["blk.0.attn_k.weight"], "blk.0.attn_k.weight");
  k_sync->set_initial_ids({"blk.0.attn_norm_out", "blk.0.attn_k.weight"});
  pimpl_->graph->add_node(k_sync);

  k_proj->set_input(k_sync->get_output(), "default");
  k_proj->set_input_names("blk.0.attn_norm_out", "blk.0.attn_k.weight");
  k_proj->set_output_name("blk.0.k_proj_out");
  k_proj->set_node_name("blk.0.k_proj");
  pimpl_->graph->add_node(k_proj);
  graph_edge_ptr k_out = k_proj->get_output("default");

  // V projection
  auto v_proj = std::make_shared<matmul_mixed_node>();
  auto v_sync = std::make_shared<result_message_sync_node>();
  v_sync->set_input(norm_out, "blk.0.attn_norm_out");
  v_sync->set_input(layer0_weights["blk.0.attn_v.weight"], "blk.0.attn_v.weight");
  v_sync->set_initial_ids({"blk.0.attn_norm_out", "blk.0.attn_v.weight"});
  pimpl_->graph->add_node(v_sync);

  v_proj->set_input(v_sync->get_output(), "default");
  v_proj->set_input_names("blk.0.attn_norm_out", "blk.0.attn_v.weight");
  v_proj->set_output_name("blk.0.v_proj_out");
  v_proj->set_node_name("blk.0.v_proj");
  pimpl_->graph->add_node(v_proj);
  graph_edge_ptr v_out = v_proj->get_output("default");

  // 2.3 Reshape Q/K for RoPE (3D → 4D → transpose → RoPE → transpose → 4D → 3D)

  // Create shape constants for reshape operations
  // Q: [batch, seq_len, 2880] → [batch, seq_len, 64, 45]
  // Use -1 for head_dim (inferred), explicit num_heads
  dynamic_tensor q_reshape_shape(dtype::int64, {4});
  auto q_reshape_shape_data = q_reshape_shape.data_ptr<int64_t>();
  q_reshape_shape_data[0] = 0;                      // Copy batch from input[0]
  q_reshape_shape_data[1] = 0;                      // Copy seq_len from input[1]
  q_reshape_shape_data[2] = pimpl_->num_q_heads;   // 64 (explicit)
  q_reshape_shape_data[3] = -1;                     // 45 (inferred from total elements)

  auto q_reshape_shape_const = std::make_shared<constant_node>(q_reshape_shape, "q_reshape_shape");
  q_reshape_shape_const->set_input(q_out, "default");  // Trigger from Q projection
  pimpl_->graph->add_node(q_reshape_shape_const);
  auto q_reshape_shape_edge = q_reshape_shape_const->get_output("default");

  // K: [batch, seq_len, 360] → [batch, seq_len, 8, 45]
  // Use -1 for head_dim (inferred), explicit num_kv_heads
  dynamic_tensor k_reshape_shape(dtype::int64, {4});
  auto k_reshape_shape_data = k_reshape_shape.data_ptr<int64_t>();
  k_reshape_shape_data[0] = 0;                      // Copy batch from input[0]
  k_reshape_shape_data[1] = 0;                      // Copy seq_len from input[1]
  k_reshape_shape_data[2] = pimpl_->num_kv_heads;  // 8 (explicit)
  k_reshape_shape_data[3] = -1;                     // 45 (inferred from total elements)

  auto k_reshape_shape_const = std::make_shared<constant_node>(k_reshape_shape, "k_reshape_shape");
  k_reshape_shape_const->set_input(k_out, "default");  // Trigger from K projection
  pimpl_->graph->add_node(k_reshape_shape_const);
  auto k_reshape_shape_edge = k_reshape_shape_const->get_output("default");

  // Reshape back: [batch, seq_len, num_heads, head_dim] → [batch, seq_len, -1]
  dynamic_tensor reshape_back_shape(dtype::int64, {3});
  auto reshape_back_shape_data = reshape_back_shape.data_ptr<int64_t>();
  reshape_back_shape_data[0] = 0;   // Copy batch
  reshape_back_shape_data[1] = 0;   // Copy seq_len
  reshape_back_shape_data[2] = -1;  // Infer from total elements

  auto reshape_back_shape_const = std::make_shared<constant_node>(reshape_back_shape, "reshape_back_shape");
  reshape_back_shape_const->set_input(q_out, "default");  // Trigger from Q projection (reused for both Q and K)
  pimpl_->graph->add_node(reshape_back_shape_const);
  auto reshape_back_shape_edge = reshape_back_shape_const->get_output("default");

  // Process Q: reshape → transpose → RoPE → transpose → reshape
  auto q_reshape_4d = std::make_shared<reshape_node>();
  auto q_reshape_4d_sync = std::make_shared<result_message_sync_node>();
  q_reshape_4d_sync->set_input(q_out, "blk.0.q_proj_out");
  q_reshape_4d_sync->set_input(q_reshape_shape_edge, "q_reshape_shape");
  q_reshape_4d_sync->set_initial_ids({"blk.0.q_proj_out", "q_reshape_shape"});
  pimpl_->graph->add_node(q_reshape_4d_sync);

  q_reshape_4d->set_input(q_reshape_4d_sync->get_output(), "default");
  q_reshape_4d->set_input_names("blk.0.q_proj_out", "q_reshape_shape");
  q_reshape_4d->set_output_name("blk.0.q_4d");
  q_reshape_4d->set_node_name("blk.0.q_reshape_4d");
  pimpl_->graph->add_node(q_reshape_4d);
  graph_edge_ptr q_4d = q_reshape_4d->get_output("default");

  // Transpose Q: [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
  auto q_transpose = std::make_shared<transpose_node>();
  q_transpose->set_perm({0, 2, 1, 3});
  q_transpose->set_input(q_4d, "default");
  q_transpose->set_input_name("blk.0.q_4d");
  q_transpose->set_output_name("blk.0.q_transposed");
  q_transpose->set_node_name("blk.0.q_transpose");
  pimpl_->graph->add_node(q_transpose);
  graph_edge_ptr q_transposed = q_transpose->get_output("default");

  // RoPE Q
  auto rope_q = std::make_shared<rope_node>();
  rope_q->set_config(layer0_head_dim, pimpl_->max_seq_len, pimpl_->rope_freq_base,
                     pimpl_->rope_scaling_factor);
  rope_q->set_input(q_transposed, "default");
  rope_q->set_input_name("blk.0.q_transposed");
  rope_q->set_output_name("blk.0.q_rope");
  rope_q->set_node_name("blk.0.rope_q");
  pimpl_->graph->add_node(rope_q);
  graph_edge_ptr q_rope = rope_q->get_output("default");

  // Transpose Q back: [batch, num_heads, seq_len, head_dim] → [batch, seq_len, num_heads, head_dim]
  auto q_transpose_back = std::make_shared<transpose_node>();
  q_transpose_back->set_perm({0, 2, 1, 3});
  q_transpose_back->set_input(q_rope, "default");
  q_transpose_back->set_input_name("blk.0.q_rope");
  q_transpose_back->set_output_name("blk.0.q_rope_4d");
  q_transpose_back->set_node_name("blk.0.q_transpose_back");
  pimpl_->graph->add_node(q_transpose_back);
  graph_edge_ptr q_rope_4d = q_transpose_back->get_output("default");

  // Reshape Q back to 3D
  auto q_reshape_3d = std::make_shared<reshape_node>();
  auto q_reshape_3d_sync = std::make_shared<result_message_sync_node>();
  q_reshape_3d_sync->set_input(q_rope_4d, "blk.0.q_rope_4d");
  q_reshape_3d_sync->set_input(reshape_back_shape_edge, "reshape_back_shape");
  q_reshape_3d_sync->set_initial_ids({"blk.0.q_rope_4d", "reshape_back_shape"});
  pimpl_->graph->add_node(q_reshape_3d_sync);

  q_reshape_3d->set_input(q_reshape_3d_sync->get_output(), "default");
  q_reshape_3d->set_input_names("blk.0.q_rope_4d", "reshape_back_shape");
  q_reshape_3d->set_output_name("blk.0.q_rope_out");
  q_reshape_3d->set_node_name("blk.0.q_reshape_3d");
  pimpl_->graph->add_node(q_reshape_3d);
  graph_edge_ptr q_rope_out = q_reshape_3d->get_output("default");

  // Process K: reshape → transpose → RoPE → transpose → reshape
  auto k_reshape_4d = std::make_shared<reshape_node>();
  auto k_reshape_4d_sync = std::make_shared<result_message_sync_node>();
  k_reshape_4d_sync->set_input(k_out, "blk.0.k_proj_out");
  k_reshape_4d_sync->set_input(k_reshape_shape_edge, "k_reshape_shape");
  k_reshape_4d_sync->set_initial_ids({"blk.0.k_proj_out", "k_reshape_shape"});
  pimpl_->graph->add_node(k_reshape_4d_sync);

  k_reshape_4d->set_input(k_reshape_4d_sync->get_output(), "default");
  k_reshape_4d->set_input_names("blk.0.k_proj_out", "k_reshape_shape");
  k_reshape_4d->set_output_name("blk.0.k_4d");
  k_reshape_4d->set_node_name("blk.0.k_reshape_4d");
  pimpl_->graph->add_node(k_reshape_4d);
  graph_edge_ptr k_4d = k_reshape_4d->get_output("default");

  // Transpose K: [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
  auto k_transpose = std::make_shared<transpose_node>();
  k_transpose->set_perm({0, 2, 1, 3});
  k_transpose->set_input(k_4d, "default");
  k_transpose->set_input_name("blk.0.k_4d");
  k_transpose->set_output_name("blk.0.k_transposed");
  k_transpose->set_node_name("blk.0.k_transpose");
  pimpl_->graph->add_node(k_transpose);
  graph_edge_ptr k_transposed = k_transpose->get_output("default");

  // RoPE K
  auto rope_k = std::make_shared<rope_node>();
  rope_k->set_config(layer0_head_dim, pimpl_->max_seq_len, pimpl_->rope_freq_base,
                     pimpl_->rope_scaling_factor);
  rope_k->set_input(k_transposed, "default");
  rope_k->set_input_name("blk.0.k_transposed");
  rope_k->set_output_name("blk.0.k_rope");
  rope_k->set_node_name("blk.0.rope_k");
  pimpl_->graph->add_node(rope_k);
  graph_edge_ptr k_rope = rope_k->get_output("default");

  // Transpose K back: [batch, num_heads, seq_len, head_dim] → [batch, seq_len, num_heads, head_dim]
  auto k_transpose_back = std::make_shared<transpose_node>();
  k_transpose_back->set_perm({0, 2, 1, 3});
  k_transpose_back->set_input(k_rope, "default");
  k_transpose_back->set_input_name("blk.0.k_rope");
  k_transpose_back->set_output_name("blk.0.k_rope_4d");
  k_transpose_back->set_node_name("blk.0.k_transpose_back");
  pimpl_->graph->add_node(k_transpose_back);
  graph_edge_ptr k_rope_4d = k_transpose_back->get_output("default");

  // Reshape K back to 3D
  auto k_reshape_3d = std::make_shared<reshape_node>();
  auto k_reshape_3d_sync = std::make_shared<result_message_sync_node>();
  k_reshape_3d_sync->set_input(k_rope_4d, "blk.0.k_rope_4d");
  k_reshape_3d_sync->set_input(reshape_back_shape_edge, "reshape_back_shape");
  k_reshape_3d_sync->set_initial_ids({"blk.0.k_rope_4d", "reshape_back_shape"});
  pimpl_->graph->add_node(k_reshape_3d_sync);

  k_reshape_3d->set_input(k_reshape_3d_sync->get_output(), "default");
  k_reshape_3d->set_input_names("blk.0.k_rope_4d", "reshape_back_shape");
  k_reshape_3d->set_output_name("blk.0.k_rope_out");
  k_reshape_3d->set_node_name("blk.0.k_reshape_3d");
  pimpl_->graph->add_node(k_reshape_3d);
  graph_edge_ptr k_rope_out = k_reshape_3d->get_output("default");

  // 2.4 Grouped Query Attention
  auto attn = std::make_shared<grouped_attention_node>();
  attn->set_config(pimpl_->num_q_heads, pimpl_->num_kv_heads, layer0_head_dim);

  auto attn_sync = std::make_shared<result_message_sync_node>();
  attn_sync->set_input(q_rope_out, "blk.0.q_rope_out");
  attn_sync->set_input(k_rope_out, "blk.0.k_rope_out");
  attn_sync->set_input(v_out, "blk.0.v_proj_out");
  attn_sync->set_initial_ids({"blk.0.q_rope_out", "blk.0.k_rope_out", "blk.0.v_proj_out"});
  pimpl_->graph->add_node(attn_sync);

  attn->set_input(attn_sync->get_output(), "default");
  attn->set_input_names({"blk.0.q_rope_out", "blk.0.k_rope_out", "blk.0.v_proj_out"});
  attn->set_output_name("blk.0.attn_out");
  attn->set_node_name("blk.0.grouped_attn");
  pimpl_->graph->add_node(attn);
  graph_edge_ptr attn_out = attn->get_output("default");

  // 2.5 Attention output projection
  auto attn_proj = std::make_shared<matmul_mixed_node>();
  auto attn_proj_sync = std::make_shared<result_message_sync_node>();
  attn_proj_sync->set_input(attn_out, "blk.0.attn_out");
  attn_proj_sync->set_input(layer0_weights["blk.0.attn_output.weight"], "blk.0.attn_output.weight");
  attn_proj_sync->set_initial_ids({"blk.0.attn_out", "blk.0.attn_output.weight"});
  pimpl_->graph->add_node(attn_proj_sync);

  attn_proj->set_input(attn_proj_sync->get_output(), "default");
  attn_proj->set_input_names("blk.0.attn_out", "blk.0.attn_output.weight");
  attn_proj->set_output_name("blk.0.attn_proj_out");
  attn_proj->set_node_name("blk.0.attn_proj");
  pimpl_->graph->add_node(attn_proj);
  graph_edge_ptr attn_proj_out = attn_proj->get_output("default");

  // 2.6 Attention residual connection
  auto attn_residual = std::make_shared<add_node>();
  auto attn_res_sync = std::make_shared<result_message_sync_node>();
  attn_res_sync->set_input(current, "embeddings");
  attn_res_sync->set_input(attn_proj_out, "blk.0.attn_proj_out");
  attn_res_sync->set_initial_ids({"embeddings", "blk.0.attn_proj_out"});
  pimpl_->graph->add_node(attn_res_sync);

  attn_residual->set_input(attn_res_sync->get_output(), "default");
  attn_residual->set_input_names("embeddings", "blk.0.attn_proj_out");
  attn_residual->set_output_name("blk.0.attn_residual_out");
  attn_residual->set_node_name("blk.0.attn_residual");
  pimpl_->graph->add_node(attn_residual);
  graph_edge_ptr attn_res_out = attn_residual->get_output("default");

  // 2.7 FFN RMSNorm (before MoE)
  auto ffn_norm = std::make_shared<rmsnorm_node>();
  ffn_norm->set_epsilon(1e-5f);

  auto ffn_norm_sync = std::make_shared<result_message_sync_node>();
  ffn_norm_sync->set_input(attn_res_out, "blk.0.attn_residual_out");
  ffn_norm_sync->set_input(layer0_weights["blk.0.post_attention_norm.weight"], "blk.0.post_attention_norm.weight");
  ffn_norm_sync->set_initial_ids({"blk.0.attn_residual_out", "blk.0.post_attention_norm.weight"});
  pimpl_->graph->add_node(ffn_norm_sync);

  ffn_norm->set_input(ffn_norm_sync->get_output(), "default");
  ffn_norm->set_input_names("blk.0.attn_residual_out", "blk.0.post_attention_norm.weight");
  ffn_norm->set_output_name("blk.0.ffn_norm_out");
  ffn_norm->set_node_name("blk.0.ffn_norm");
  pimpl_->graph->add_node(ffn_norm);
  graph_edge_ptr ffn_norm_out = ffn_norm->get_output("default");

  // 2.8 MoE Router (top-k expert selection)
  auto router = std::make_shared<moe_router_node>();
  router->set_config(pimpl_->num_experts, pimpl_->expert_top_k);

  auto router_sync = std::make_shared<result_message_sync_node>();
  router_sync->set_input(ffn_norm_out, "blk.0.ffn_norm_out");
  router_sync->set_input(layer0_weights["blk.0.ffn_gate_inp.weight"], "blk.0.ffn_gate_inp.weight");
  router_sync->set_initial_ids({"blk.0.ffn_norm_out", "blk.0.ffn_gate_inp.weight"});
  pimpl_->graph->add_node(router_sync);

  router->set_input(router_sync->get_output(), "default");
  router->set_input_names("blk.0.ffn_norm_out", "blk.0.ffn_gate_inp.weight");
  router->set_output_name("blk.0.router_out");
  router->set_node_name("blk.0.moe_router");
  pimpl_->graph->add_node(router);
  graph_edge_ptr router_out = router->get_output("default");

  // 2.9 Expert MLPs (32 experts in parallel)
  std::vector<graph_edge_ptr> expert_outputs;
  for (int expert_id = 0; expert_id < pimpl_->num_experts; ++expert_id) {
    auto expert = std::make_shared<expert_mlp_node>(expert_id);

    auto expert_sync = std::make_shared<result_message_sync_node>();
    expert_sync->set_input(ffn_norm_out, "blk.0.ffn_norm_out");
    expert_sync->set_input(layer0_weights["blk.0.ffn_up_exps.weight"], "blk.0.ffn_up_exps.weight");
    expert_sync->set_input(layer0_weights["blk.0.ffn_gate_exps.weight"], "blk.0.ffn_gate_exps.weight");
    expert_sync->set_input(layer0_weights["blk.0.ffn_down_exps.weight"], "blk.0.ffn_down_exps.weight");
    expert_sync->set_initial_ids({"blk.0.ffn_norm_out", "blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"});
    pimpl_->graph->add_node(expert_sync);

    expert->set_input(expert_sync->get_output(), "default");
    expert->set_input_names({"blk.0.ffn_norm_out", "blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"});
    expert->set_output_name("blk.0.expert_" + std::to_string(expert_id) + "_out");
    expert->set_node_name("blk.0.expert_" + std::to_string(expert_id));
    pimpl_->graph->add_node(expert);
    expert_outputs.push_back(expert->get_output("default"));
  }

  // 2.10 Expert merge (weighted sum of top-k experts)
  auto expert_merge = std::make_shared<expert_merge_node>();
  expert_merge->set_config(pimpl_->num_experts, pimpl_->expert_top_k);

  std::vector<std::string> merge_input_names = {"blk.0.router_out"};
  for (int i = 0; i < pimpl_->num_experts; ++i) {
    merge_input_names.push_back("blk.0.expert_" + std::to_string(i) + "_out");
  }

  auto merge_sync = std::make_shared<result_message_sync_node>();
  merge_sync->set_input(router_out, "blk.0.router_out");
  for (int i = 0; i < pimpl_->num_experts; ++i) {
    merge_sync->set_input(expert_outputs[i], "blk.0.expert_" + std::to_string(i) + "_out");
  }
  merge_sync->set_initial_ids(merge_input_names);
  pimpl_->graph->add_node(merge_sync);

  expert_merge->set_input(merge_sync->get_output(), "default");
  expert_merge->set_input_names(merge_input_names);
  expert_merge->set_output_name("blk.0.expert_merge_out");
  expert_merge->set_node_name("blk.0.expert_merge");
  pimpl_->graph->add_node(expert_merge);
  graph_edge_ptr merge_out = expert_merge->get_output("default");

  // 2.11 FFN residual connection
  auto ffn_residual = std::make_shared<add_node>();
  auto ffn_res_sync = std::make_shared<result_message_sync_node>();
  ffn_res_sync->set_input(attn_res_out, "blk.0.attn_residual_out");
  ffn_res_sync->set_input(merge_out, "blk.0.expert_merge_out");
  ffn_res_sync->set_initial_ids({"blk.0.attn_residual_out", "blk.0.expert_merge_out"});
  pimpl_->graph->add_node(ffn_res_sync);

  ffn_residual->set_input(ffn_res_sync->get_output(), "default");
  ffn_residual->set_input_names("blk.0.attn_residual_out", "blk.0.expert_merge_out");
  ffn_residual->set_output_name("blk.0.ffn_residual_out");
  ffn_residual->set_node_name("blk.0.ffn_residual");
  pimpl_->graph->add_node(ffn_residual);

  current = ffn_residual->get_output("default");

  // ========== 3. Final RMSNorm ==========
  auto final_norm = std::make_shared<rmsnorm_node>();
  final_norm->set_epsilon(1e-5f);

  auto final_norm_sync = std::make_shared<result_message_sync_node>();
  final_norm_sync->set_input(current, "blk.0.ffn_residual_out");
  final_norm_sync->set_input(output_norm_weight_edge, "output_norm.weight");
  final_norm_sync->set_initial_ids({"blk.0.ffn_residual_out", "output_norm.weight"});
  pimpl_->graph->add_node(final_norm_sync);

  final_norm->set_input(final_norm_sync->get_output(), "default");
  final_norm->set_input_names("blk.0.ffn_residual_out", "output_norm.weight");
  final_norm->set_output_name("final_norm_out");
  final_norm->set_node_name("final_norm");
  pimpl_->graph->add_node(final_norm);
  graph_edge_ptr final_norm_out = final_norm->get_output("default");

  // ========== 4. Output projection ==========
  auto output_proj = std::make_shared<matmul_mixed_node>();

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

  // ========== 5. Output sync and I/O nodes ==========
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

  std::cout << "✓ Single-layer test graph built (Layer 0 with full transformer)\n";
}

void gpt_oss_engine::wire_io_nodes(graph_edge_ptr input_placeholder, graph_edge_ptr logits_output) {
  // Create I/O nodes
  pimpl_->input_node = std::make_shared<model_input_node>();
  pimpl_->output_node = std::make_shared<model_output_node>();

  // Create extractor to extract input_ids from input_node
  auto extractor = std::make_shared<result_field_extractor_node>();
  pimpl_->graph->add_node(extractor);
  extractor->set_input(pimpl_->input_node->get_output(), "default");

  // Add output for input_ids
  auto input_ids_out = extractor->add_output("input_ids");

  // Replace all references to placeholder edge with extractor output
  for (uint32_t i = 0; i < pimpl_->graph->get_node_count(); ++i) {
    auto node = pimpl_->graph->get_node(i);
    const auto& inputs = node->get_inputs();
    for (const auto& [port_name, input_edge] : inputs) {
      if (input_edge == input_placeholder) {
        node->set_input(input_ids_out, port_name);
      }
    }
  }

  // Connect constant_nodes to frame source (onnx_importer pattern lines 697-702)
  // This allows constant_nodes to receive frame numbers for message sending
  if (!pimpl_->constant_nodes.empty()) {
    for (auto& const_node : pimpl_->constant_nodes) {
      const_node->set_input(input_ids_out, "default");
    }
  }

  // Create sync node for outputs
  auto output_sync = std::make_shared<result_message_sync_node>();
  pimpl_->graph->add_node(output_sync);
  output_sync->set_input(logits_output, "logits");
  output_sync->set_initial_ids({"logits"});

  // Connect sync output to output_node
  pimpl_->output_node->set_input(output_sync->get_output(), "default");

  // Add I/O nodes to graph
  pimpl_->graph->add_node(pimpl_->input_node);
  pimpl_->graph->add_node(pimpl_->output_node);

  // Deploy graph to proc
  pimpl_->proc = std::make_unique<graph_proc>();
  pimpl_->proc->deploy(pimpl_->graph);
}

std::string gpt_oss_engine::generate(const std::string& prompt, size_t max_tokens,
                                     float temperature) {
  if (!pimpl_->loaded) {
    std::cerr << "Error: Model not loaded\n";
    return "";
  }

  // Tokenize prompt
  std::vector<uint32_t> tokens = pimpl_->tokenizer->encode(prompt);

  std::string generated_text;

  // Set up output callback to collect results
  std::unordered_map<std::string, dynamic_tensor> outputs;
  bool output_received = false;

  pimpl_->output_node->set_callback(
      [&outputs, &output_received](const std::unordered_map<std::string, dynamic_tensor>& result) {
        outputs = result;
        output_received = true;
      });

  // Generation loop
  for (size_t step = 0; step < max_tokens; ++step) {
    output_received = false;

    // Create input tensor: [1, seq_len]
    std::vector<int64_t> shape = {1, static_cast<int64_t>(tokens.size())};
    dynamic_tensor input_tensor(dtype::int32, shape);
    int32_t* data = input_tensor.data_ptr<int32_t>();
    for (size_t j = 0; j < tokens.size(); ++j) {
      data[j] = static_cast<int32_t>(tokens[j]);
    }

    // Set input tensor
    pimpl_->input_node->set_tensor("input_ids", input_tensor);
    pimpl_->input_node->set_frame_number(step + 1);

    // Run inference (graph_proc.run() executes the entire graph synchronously)
    pimpl_->proc->run();

    if (!output_received) {
      std::cerr << "Warning: No output received for step " << step << "\n";
      break;
    }

    // Get logits
    auto it = outputs.find("logits");
    if (it == outputs.end()) {
      std::cerr << "Error: No logits output\n";
      break;
    }

    const auto& logits = it->second;
    const float* logits_data = logits.data_ptr<float>();

    // Get last position logits for next token prediction
    // logits shape: [1, seq_len, vocab_size]
    int64_t seq_len = logits.dim(1);
    int64_t vocab_size = logits.dim(2);
    int64_t last_pos = seq_len - 1;
    const float* last_logits = logits_data + last_pos * vocab_size;

    // Sample next token
    uint32_t next_token = sample_token(last_logits, vocab_size, temperature);

    // Check for EOS token
    if (next_token == pimpl_->tokenizer->eos_token_id()) {
      std::cout << "[EOS]\n";
      break;
    }

    // Add token to sequence
    tokens.push_back(next_token);

    // Decode and output
    std::string piece = pimpl_->tokenizer->decode({next_token});
    std::cout << piece << std::flush;
    generated_text += piece;
  }

  std::cout << "\n";
  return generated_text;
}

uint32_t gpt_oss_engine::sample_token(const float* logits, int64_t vocab_size, float temperature) {
  if (temperature < 1e-6f) {
    // Greedy sampling
    return static_cast<uint32_t>(
        std::distance(logits, std::max_element(logits, logits + vocab_size)));
  }

  // Temperature sampling
  std::vector<float> probs(vocab_size);
  float max_logit = *std::max_element(logits, logits + vocab_size);

  // Softmax with temperature
  float sum = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) / temperature);
    sum += probs[i];
  }

  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] /= sum;
  }

  // Sample from distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probs.begin(), probs.end());

  return static_cast<uint32_t>(dist(gen));
}

bool gpt_oss_engine::is_loaded() const { return pimpl_->loaded; }

int64_t gpt_oss_engine::get_vocab_size() const { return pimpl_->vocab_size; }

int64_t gpt_oss_engine::get_num_layers() const { return pimpl_->num_layers; }

int64_t gpt_oss_engine::get_hidden_dim() const { return pimpl_->hidden_dim; }

}  // namespace coalsack
