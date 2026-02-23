#include "coalsack/llm/vision_encoder.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/core/graph_proc.h"
#include "coalsack/gguf/gguf_dequant.h"
#include "coalsack/gguf/gguf_loader.h"
#include "coalsack/nn/model_io_nodes.h"
#include "coalsack/nn/nn_nodes.h"
#include "coalsack/nn/nn_ops/add_node.h"
#include "coalsack/nn/nn_ops/concat_node.h"
#include "coalsack/nn/nn_ops/gelu_node.h"
#include "coalsack/nn/nn_ops/layer_normalization_node.h"
#include "coalsack/nn/nn_ops/matmul_transpose_mixed_node.h"
#include "coalsack/nn/nn_ops/pixel_shuffle_node.h"
#include "coalsack/nn/nn_ops/rope_2d_node.h"
#include "coalsack/nn/nn_ops/sdp_attention_node.h"
#include "coalsack/nn/result_message_nodes.h"

namespace coalsack {

namespace {

// Load a tensor from a GGUF file.
// GGUF stores dimensions as ne[0]=fastest (innermost), which is the reverse of
// our row-major convention.  This function reverses the dimension order so the
// returned dynamic_tensor has shape [outer, ..., inner] (row-major).
// Returns float32 or float16 tensor; any other quantised type is dequantised to float32.
static dynamic_tensor load_gguf_tensor(const gguf_loader& loader, const std::string& name,
                                       std::ifstream& file) {
  auto info_opt = loader.get_tensor_info(name);
  if (!info_opt) {
    throw std::runtime_error("vision_encoder: tensor not found: " + name);
  }
  const auto& info = *info_opt;

  std::vector<int64_t> shape;
  for (int i = static_cast<int>(info.shape.size()) - 1; i >= 0; --i) {
    shape.push_back(static_cast<int64_t>(info.shape[i]));
  }
  int64_t numel = 1;
  for (auto d : shape) numel *= d;

  file.seekg(info.offset);
  std::vector<uint8_t> raw(info.size);
  file.read(reinterpret_cast<char*>(raw.data()), info.size);

  if (info.type == ggml_type::F32) {
    dynamic_tensor t(dtype::float32, shape);
    std::memcpy(t.data_ptr<float>(), raw.data(), info.size);
    return t;
  } else if (info.type == ggml_type::F16) {
    dynamic_tensor t(dtype::float16, shape);
    std::memcpy(t.data_ptr<uint16_t>(), raw.data(), info.size);
    return t;
  } else {
    dynamic_tensor t(dtype::float32, shape);
    if (!dequantize_tensor(raw.data(), t.data_ptr<float>(), numel, info.type)) {
      throw std::runtime_error("vision_encoder: unsupported quant type for: " + name);
    }
    return t;
  }
}

// Graph builder — encapsulates all sync+op node pair creation patterns for the ViT graph.
class vit_graph_builder {
 public:
  vit_graph_builder(std::shared_ptr<subgraph> graph, float ln_eps)
      : graph_(std::move(graph)), ln_eps_(ln_eps) {}

  // sync + matmul_transpose_mixed_node
  // Computes: A[*,M,K] @ B[*,N,K]^T -> [*,M,N]
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

  // sync + add_node (supports broadcasting for bias adds)
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

  // sync(3) + layer_normalization_node (X, scale, bias)
  graph_edge_ptr make_layernorm(graph_edge_ptr in, const std::string& in_name, graph_edge_ptr scale,
                                const std::string& scale_name, graph_edge_ptr bias,
                                const std::string& bias_name, const std::string& out_name,
                                const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in, in_name);
    sync->set_input(scale, scale_name);
    sync->set_input(bias, bias_name);
    sync->set_initial_ids({in_name, scale_name, bias_name});
    graph_->add_node(sync);

    auto node = std::make_shared<layer_normalization_node>();
    node->set_epsilon(ln_eps_);
    node->set_input(sync->get_output(), "default");
    node->set_input_names({in_name, scale_name, bias_name});
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync(2) + concat_node
  graph_edge_ptr make_concat(graph_edge_ptr in1, const std::string& name1, graph_edge_ptr in2,
                             const std::string& name2, int64_t axis, const std::string& out_name,
                             const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(in1, name1);
    sync->set_input(in2, name2);
    sync->set_initial_ids({name1, name2});
    graph_->add_node(sync);

    auto node = std::make_shared<concat_node>(2);
    node->set_axis(axis);
    node->set_input(sync->get_output(), "default");
    node->set_input_names({name1, name2});
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // gelu_node (unary, no sync required)
  graph_edge_ptr make_gelu(graph_edge_ptr in, const std::string& in_name,
                           const std::string& out_name, const std::string& node_name) {
    auto node = std::make_shared<gelu_node>();
    node->set_input(in, "default");
    node->set_input_name(in_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // rope_2d_node (unary, no sync required)
  graph_edge_ptr make_rope2d(graph_edge_ptr in, const std::string& in_name,
                             const std::vector<int>& pos_w, const std::vector<int>& pos_h,
                             float theta, int n_head, int d_head, const std::string& out_name,
                             const std::string& node_name) {
    auto node = std::make_shared<rope_2d_node>();
    node->set_positions(pos_w, pos_h);
    node->set_theta(theta);
    node->set_num_heads(n_head);
    node->set_head_dim(d_head);
    node->set_input(in, "default");
    node->set_input_name(in_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // sync(3) + sdp_attention_node (Q, K, V)
  graph_edge_ptr make_sdp_attn(graph_edge_ptr q, const std::string& q_name, graph_edge_ptr k,
                               const std::string& k_name, graph_edge_ptr v,
                               const std::string& v_name, int n_head, int d_head,
                               const std::string& out_name, const std::string& node_name) {
    auto sync = std::make_shared<result_message_sync_node>();
    sync->set_input(q, q_name);
    sync->set_input(k, k_name);
    sync->set_input(v, v_name);
    sync->set_initial_ids({q_name, k_name, v_name});
    graph_->add_node(sync);

    auto node = std::make_shared<sdp_attention_node>();
    node->set_num_heads(n_head);
    node->set_head_dim(d_head);
    node->set_input(sync->get_output(), "default");
    node->set_input_names({q_name, k_name, v_name});
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // pixel_shuffle_node (unary, no sync required)
  graph_edge_ptr make_pixel_shuffle(graph_edge_ptr in, const std::string& in_name, int scale,
                                    int patches_per_row, const std::string& out_name,
                                    const std::string& node_name) {
    auto node = std::make_shared<pixel_shuffle_node>();
    node->set_scale_factor(scale);
    node->set_patches_per_row(patches_per_row);
    node->set_input(in, "default");
    node->set_input_name(in_name);
    node->set_output_name(out_name);
    node->set_node_name(node_name);
    graph_->add_node(node);
    return node->get_output("default");
  }

  // layer_scheduler_node -- passthrough, breaks synchronous call-stack depth
  // between transformer layers (same approach as llm_engine).
  graph_edge_ptr make_layer_scheduler(graph_edge_ptr in) {
    auto node = std::make_shared<layer_scheduler_node>();
    node->set_input(in, "default");
    graph_->add_node(node);
    return node->get_output();
  }

 private:
  std::shared_ptr<subgraph> graph_;
  float ln_eps_;
};

}  // namespace

struct vision_encoder::impl {
  config cfg;
  bool loaded = false;

  // All weight tensors (weight name -> dynamic_tensor, f32 or f16)
  std::unordered_map<std::string, dynamic_tensor> weights;

  // Graph and execution engine
  std::shared_ptr<subgraph> graph;
  std::unique_ptr<graph_proc> proc;
  std::shared_ptr<model_source_node> input_node;
  std::shared_ptr<model_output_node> output_node;

  void build_graph(const std::vector<int>& pos_w, const std::vector<int>& pos_h);
};

vision_encoder::vision_encoder() : pimpl_(std::make_unique<impl>()) {}
vision_encoder::~vision_encoder() = default;

bool vision_encoder::is_loaded() const { return pimpl_->loaded; }
const vision_encoder::config& vision_encoder::get_config() const { return pimpl_->cfg; }

int vision_encoder::n_output_tokens() const {
  const auto& c = pimpl_->cfg;
  int n_patches = (c.image_size / c.patch_size) * (c.image_size / c.patch_size);
  return n_patches / (c.scale_factor * c.scale_factor);
}

bool vision_encoder::load(const std::string& mmproj_path) {
  gguf_loader loader;
  if (!loader.load(mmproj_path)) {
    spdlog::error("vision_encoder: failed to load GGUF: {}", mmproj_path);
    return false;
  }

  auto& cfg = pimpl_->cfg;
  if (auto v = loader.get_uint32("clip.vision.image_size")) cfg.image_size = *v;
  if (auto v = loader.get_uint32("clip.vision.patch_size")) cfg.patch_size = *v;
  if (auto v = loader.get_uint32("clip.vision.embedding_length")) cfg.n_embd = *v;
  if (auto v = loader.get_uint32("clip.vision.feed_forward_length")) cfg.ffn_dim = *v;
  if (auto v = loader.get_uint32("clip.vision.block_count")) cfg.n_layer = *v;
  if (auto v = loader.get_uint32("clip.vision.attention.head_count")) cfg.n_head = *v;
  if (auto v = loader.get_uint32("clip.vision.projection_dim")) cfg.proj_dim = *v;
  if (auto v = loader.get_uint32("clip.vision.projector.scale_factor")) cfg.scale_factor = *v;
  if (auto v = loader.get_float32("clip.vision.image_mean")) cfg.image_mean = *v;
  if (auto v = loader.get_float32("clip.vision.image_std")) cfg.image_std = *v;
  if (auto v = loader.get_float32("clip.vision.attention.layer_norm_epsilon")) cfg.ln_eps = *v;

  // Infer projector intermediate dimension from mm.model.mlp.1.weight.
  // GGUF ne[0]=in_features, ne[1]=out_features => proj_intermediate = out_features.
  if (auto info = loader.get_tensor_info("mm.model.mlp.1.weight")) {
    cfg.proj_intermediate = static_cast<int>(info->shape[1]);
  }

  spdlog::info(
      "vision_encoder: image={}x{} patch={} embd={} heads={} layers={} ffn={} proj={} "
      "scale={} proj_mid={}",
      cfg.image_size, cfg.image_size, cfg.patch_size, cfg.n_embd, cfg.n_head, cfg.n_layer,
      cfg.ffn_dim, cfg.proj_dim, cfg.scale_factor, cfg.proj_intermediate);

  std::ifstream file(mmproj_path, std::ios::binary);
  if (!file) {
    spdlog::error("vision_encoder: cannot open file: {}", mmproj_path);
    return false;
  }

  auto load_w = [&](const std::string& name) -> dynamic_tensor {
    return load_gguf_tensor(loader, name, file);
  };

  auto& wt = pimpl_->weights;

  // Global ViT weights.
  // class_embd is [n_embd] in GGUF; reshape to [1, n_embd] for axis-0 concat.
  wt["v.patch_embd.weight"] = load_w("v.patch_embd.weight");
  wt["v.class_embd"] = load_w("v.class_embd").reshape({1, cfg.n_embd});
  wt["v.position_embd.weight"] = load_w("v.position_embd.weight");
  wt["v.pre_ln.weight"] = load_w("v.pre_ln.weight");
  wt["v.pre_ln.bias"] = load_w("v.pre_ln.bias");
  wt["v.post_ln.weight"] = load_w("v.post_ln.weight");
  wt["v.post_ln.bias"] = load_w("v.post_ln.bias");

  // Projector MLP weights (no bias).
  wt["mm.model.mlp.1.weight"] = load_w("mm.model.mlp.1.weight");
  wt["mm.model.mlp.2.weight"] = load_w("mm.model.mlp.2.weight");
  wt["mm.model.fc.weight"] = load_w("mm.model.fc.weight");

  // Per-layer weights.
  for (int l = 0; l < cfg.n_layer; ++l) {
    const std::string p = "v.blk." + std::to_string(l);
    wt[p + ".ln1.weight"] = load_w(p + ".ln1.weight");
    wt[p + ".ln1.bias"] = load_w(p + ".ln1.bias");
    wt[p + ".ln2.weight"] = load_w(p + ".ln2.weight");
    wt[p + ".ln2.bias"] = load_w(p + ".ln2.bias");
    wt[p + ".attn_q.weight"] = load_w(p + ".attn_q.weight");
    wt[p + ".attn_q.bias"] = load_w(p + ".attn_q.bias");
    wt[p + ".attn_k.weight"] = load_w(p + ".attn_k.weight");
    wt[p + ".attn_k.bias"] = load_w(p + ".attn_k.bias");
    wt[p + ".attn_v.weight"] = load_w(p + ".attn_v.weight");
    wt[p + ".attn_v.bias"] = load_w(p + ".attn_v.bias");
    wt[p + ".attn_out.weight"] = load_w(p + ".attn_out.weight");
    wt[p + ".attn_out.bias"] = load_w(p + ".attn_out.bias");
    wt[p + ".ffn_up.weight"] = load_w(p + ".ffn_up.weight");
    wt[p + ".ffn_up.bias"] = load_w(p + ".ffn_up.bias");
    wt[p + ".ffn_down.weight"] = load_w(p + ".ffn_down.weight");
    wt[p + ".ffn_down.bias"] = load_w(p + ".ffn_down.bias");
  }

  // Precompute 2D position IDs.
  // patch i at row (i/prw), col (i%prw):  pos_w[i]=col+1, pos_h[i]=row+1.
  // CLS token is the last position (index n_patches): pos_w = pos_h = 0.
  const int prw = cfg.image_size / cfg.patch_size;
  const int n_patches = prw * prw;
  const int n_pos = n_patches + 1;
  std::vector<int> pos_w(n_pos, 0), pos_h(n_pos, 0);
  for (int i = 0; i < n_patches; ++i) {
    pos_w[i] = (i % prw) + 1;
    pos_h[i] = (i / prw) + 1;
  }

  pimpl_->build_graph(pos_w, pos_h);

  pimpl_->loaded = true;
  spdlog::info("vision_encoder: loaded {} layers, {} output tokens", cfg.n_layer,
               n_output_tokens());
  return true;
}

void vision_encoder::impl::build_graph(const std::vector<int>& pos_w,
                                       const std::vector<int>& pos_h) {
  graph = std::make_shared<subgraph>();
  input_node = std::make_shared<model_source_node>();
  output_node = std::make_shared<model_output_node>();

  // Register all weight tensors with the input node.
  // "patches_flat" is set per encode() call in encode_f32() before push().
  for (const auto& [name, tensor] : weights) {
    input_node->set_tensor(name, tensor);
  }

  // Fan out every tensor field from the input message to individual edges.
  auto extractor = std::make_shared<result_field_extractor_node>();
  graph->add_node(extractor);
  extractor->set_input(input_node->get_output(), "default");

  auto e = [&](const std::string& name) -> graph_edge_ptr { return extractor->add_output(name); };

  // Global weight edges.
  auto patches_flat_e = e("patches_flat");
  auto patch_embd_w_e = e("v.patch_embd.weight");
  auto class_embd_e = e("v.class_embd");
  auto pos_embd_e = e("v.position_embd.weight");
  auto pre_ln_w_e = e("v.pre_ln.weight");
  auto pre_ln_b_e = e("v.pre_ln.bias");
  auto post_ln_w_e = e("v.post_ln.weight");
  auto post_ln_b_e = e("v.post_ln.bias");
  auto mm_mlp1_w_e = e("mm.model.mlp.1.weight");
  auto mm_mlp2_w_e = e("mm.model.mlp.2.weight");
  auto mm_fc_w_e = e("mm.model.fc.weight");

  // Per-layer weight edges.
  struct layer_edges_t {
    graph_edge_ptr ln1_w, ln1_b, ln2_w, ln2_b;
    graph_edge_ptr q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b;
    graph_edge_ptr ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b;
  };
  std::vector<layer_edges_t> le(cfg.n_layer);
  for (int l = 0; l < cfg.n_layer; ++l) {
    const std::string p = "v.blk." + std::to_string(l);
    le[l].ln1_w = e(p + ".ln1.weight");
    le[l].ln1_b = e(p + ".ln1.bias");
    le[l].ln2_w = e(p + ".ln2.weight");
    le[l].ln2_b = e(p + ".ln2.bias");
    le[l].q_w = e(p + ".attn_q.weight");
    le[l].q_b = e(p + ".attn_q.bias");
    le[l].k_w = e(p + ".attn_k.weight");
    le[l].k_b = e(p + ".attn_k.bias");
    le[l].v_w = e(p + ".attn_v.weight");
    le[l].v_b = e(p + ".attn_v.bias");
    le[l].out_w = e(p + ".attn_out.weight");
    le[l].out_b = e(p + ".attn_out.bias");
    le[l].ffn_up_w = e(p + ".ffn_up.weight");
    le[l].ffn_up_b = e(p + ".ffn_up.bias");
    le[l].ffn_down_w = e(p + ".ffn_down.weight");
    le[l].ffn_down_b = e(p + ".ffn_down.bias");
  }

  vit_graph_builder gb(graph, cfg.ln_eps);
  const int n_head = cfg.n_head;
  const int d_head = cfg.n_embd / cfg.n_head;

  // Patch embedding: patches_flat [n_patches, patch_dim] @ patch_embd_w [n_embd, patch_dim]^T ->
  // [n_patches, n_embd]
  auto cur_e = gb.make_matmul(patches_flat_e, "patches_flat", patch_embd_w_e, "v.patch_embd.weight",
                              "patch_embd_out", "patch_embd");
  std::string cur_name = "patch_embd_out";

  // Concatenate CLS token [1, n_embd] at end -> [n_pos, n_embd]
  cur_e = gb.make_concat(cur_e, cur_name, class_embd_e, "v.class_embd", 0, "tokens_with_cls",
                         "cls_concat");
  cur_name = "tokens_with_cls";

  // Add learned position embeddings
  cur_e =
      gb.make_add(cur_e, cur_name, pos_embd_e, "v.position_embd.weight", "tokens_pos", "pos_add");
  cur_name = "tokens_pos";

  // Pre-LayerNorm
  cur_e = gb.make_layernorm(cur_e, cur_name, pre_ln_w_e, "v.pre_ln.weight", pre_ln_b_e,
                            "v.pre_ln.bias", "pre_ln_out", "pre_ln");
  cur_name = "pre_ln_out";

  // Transformer layers
  for (int l = 0; l < cfg.n_layer; ++l) {
    // Break synchronous call-stack depth between layers.
    cur_e = gb.make_layer_scheduler(cur_e);
    // cur_name is unchanged: layer_scheduler passes messages through unmodified.

    const std::string ls = "l" + std::to_string(l);       // short layer id
    const std::string lp = "v.blk." + std::to_string(l);  // GGUF weight prefix

    // LayerNorm 1
    auto ln1_e = gb.make_layernorm(cur_e, cur_name, le[l].ln1_w, lp + ".ln1.weight", le[l].ln1_b,
                                   lp + ".ln1.bias", ls + "_ln1", "ln1_" + ls);
    const std::string ln1_name = ls + "_ln1";

    // Q projection + bias
    auto q_e = gb.make_matmul(ln1_e, ln1_name, le[l].q_w, lp + ".attn_q.weight", ls + "_q_proj",
                              "q_proj_" + ls);
    q_e =
        gb.make_add(q_e, ls + "_q_proj", le[l].q_b, lp + ".attn_q.bias", ls + "_q", "q_bias_" + ls);

    // K projection + bias
    auto k_e = gb.make_matmul(ln1_e, ln1_name, le[l].k_w, lp + ".attn_k.weight", ls + "_k_proj",
                              "k_proj_" + ls);
    k_e =
        gb.make_add(k_e, ls + "_k_proj", le[l].k_b, lp + ".attn_k.bias", ls + "_k", "k_bias_" + ls);

    // V projection + bias
    auto v_e = gb.make_matmul(ln1_e, ln1_name, le[l].v_w, lp + ".attn_v.weight", ls + "_v_proj",
                              "v_proj_" + ls);
    v_e =
        gb.make_add(v_e, ls + "_v_proj", le[l].v_b, lp + ".attn_v.bias", ls + "_v", "v_bias_" + ls);

    // 2D RoPE on Q and K
    q_e = gb.make_rope2d(q_e, ls + "_q", pos_w, pos_h, cfg.rope_theta, n_head, d_head,
                         ls + "_q_rope", "q_rope_" + ls);
    k_e = gb.make_rope2d(k_e, ls + "_k", pos_w, pos_h, cfg.rope_theta, n_head, d_head,
                         ls + "_k_rope", "k_rope_" + ls);

    // Scaled dot-product attention
    auto attn_e = gb.make_sdp_attn(q_e, ls + "_q_rope", k_e, ls + "_k_rope", v_e, ls + "_v", n_head,
                                   d_head, ls + "_attn_out", "sdp_" + ls);

    // Output projection + bias
    attn_e = gb.make_matmul(attn_e, ls + "_attn_out", le[l].out_w, lp + ".attn_out.weight",
                            ls + "_out_proj", "out_proj_" + ls);
    attn_e = gb.make_add(attn_e, ls + "_out_proj", le[l].out_b, lp + ".attn_out.bias",
                         ls + "_attn_final", "out_bias_" + ls);

    // Residual 1: tokens += attn_out
    cur_e = gb.make_add(cur_e, cur_name, attn_e, ls + "_attn_final", ls + "_res1", "res1_" + ls);
    cur_name = ls + "_res1";

    // LayerNorm 2
    auto ln2_e = gb.make_layernorm(cur_e, cur_name, le[l].ln2_w, lp + ".ln2.weight", le[l].ln2_b,
                                   lp + ".ln2.bias", ls + "_ln2", "ln2_" + ls);
    const std::string ln2_name = ls + "_ln2";

    // FFN: up projection + bias + GELU
    auto up_e = gb.make_matmul(ln2_e, ln2_name, le[l].ffn_up_w, lp + ".ffn_up.weight",
                               ls + "_ffn_up_proj", "ffn_up_proj_" + ls);
    up_e = gb.make_add(up_e, ls + "_ffn_up_proj", le[l].ffn_up_b, lp + ".ffn_up.bias",
                       ls + "_ffn_up", "ffn_up_bias_" + ls);
    up_e = gb.make_gelu(up_e, ls + "_ffn_up", ls + "_ffn_gelu", "ffn_gelu_" + ls);

    // FFN: down projection + bias
    auto down_e = gb.make_matmul(up_e, ls + "_ffn_gelu", le[l].ffn_down_w, lp + ".ffn_down.weight",
                                 ls + "_ffn_down_proj", "ffn_down_proj_" + ls);
    down_e = gb.make_add(down_e, ls + "_ffn_down_proj", le[l].ffn_down_b, lp + ".ffn_down.bias",
                         ls + "_ffn_down", "ffn_down_bias_" + ls);

    // Residual 2: tokens += ffn_out
    cur_e = gb.make_add(cur_e, cur_name, down_e, ls + "_ffn_down", ls + "_res2", "res2_" + ls);
    cur_name = ls + "_res2";
  }

  // Post-LayerNorm
  cur_e = gb.make_layernorm(cur_e, cur_name, post_ln_w_e, "v.post_ln.weight", post_ln_b_e,
                            "v.post_ln.bias", "post_ln_out", "post_ln");
  cur_name = "post_ln_out";

  // Pixel shuffle: drops CLS token internally, output [n_merged, merged_embd]
  const int prw = cfg.image_size / cfg.patch_size;
  cur_e = gb.make_pixel_shuffle(cur_e, cur_name, cfg.scale_factor, prw, "pixel_shuffle_out",
                                "pixel_shuffle");
  cur_name = "pixel_shuffle_out";

  // Projector MLP (no bias)
  cur_e = gb.make_matmul(cur_e, cur_name, mm_mlp1_w_e, "mm.model.mlp.1.weight", "proj1_out",
                         "proj_mlp1");
  cur_e = gb.make_gelu(cur_e, "proj1_out", "proj1_gelu", "proj_gelu1");

  // mlp2: [n_merged, proj_mid] @ [proj_mid, proj_mid]^T -> [n_merged, proj_mid]
  cur_e = gb.make_matmul(cur_e, "proj1_gelu", mm_mlp2_w_e, "mm.model.mlp.2.weight", "proj2_out",
                         "proj_mlp2");
  cur_e = gb.make_gelu(cur_e, "proj2_out", "proj2_gelu", "proj_gelu2");

  // fc: [n_merged, proj_mid] @ [proj_dim, proj_mid]^T -> [n_merged, proj_dim]
  cur_e = gb.make_matmul(cur_e, "proj2_gelu", mm_fc_w_e, "mm.model.fc.weight", "vision_output",
                         "proj_fc");

  // Connect to output node and add I/O nodes to graph.
  output_node->set_input(cur_e, "vision_output");

  graph->add_node(input_node);
  graph->add_node(output_node);

  proc = std::make_unique<graph_proc>();
  proc->deploy(graph);
}

dynamic_tensor vision_encoder::encode(const uint8_t* data, int width, int height) {
  const auto& cfg = pimpl_->cfg;
  if (width != cfg.image_size || height != cfg.image_size) {
    throw std::runtime_error("vision_encoder::encode: image size mismatch");
  }
  int N = width * height * 3;
  std::vector<float> f32(N);
  for (int i = 0; i < N; ++i) {
    f32[i] = (data[i] / 255.0f - cfg.image_mean) / cfg.image_std;
  }
  return encode_f32(f32.data(), width, height);
}

dynamic_tensor vision_encoder::encode_f32(const float* image_data, int width, int height) {
  if (!pimpl_->loaded) throw std::runtime_error("vision_encoder not loaded");

  const auto& cfg = pimpl_->cfg;
  const int W = cfg.image_size;
  const int H = cfg.image_size;
  const int P = cfg.patch_size;
  const int prw = W / P;            // patches per row (24)
  const int n_patches = prw * prw;  // 576
  const int patch_dim = P * P * 3;  // 588

  if (width != W || height != H) {
    throw std::runtime_error("vision_encoder::encode_f32: image size mismatch");
  }

  // Unfold image into patches: [n_patches, patch_dim]
  dynamic_tensor patches(dtype::float32,
                         {static_cast<int64_t>(n_patches), static_cast<int64_t>(patch_dim)});
  float* pd = patches.data_ptr<float>();
  for (int py = 0; py < prw; ++py) {
    for (int px = 0; px < prw; ++px) {
      int pi = py * prw + px;
      float* dst = pd + pi * patch_dim;
      for (int ky = 0; ky < P; ++ky) {
        for (int kx = 0; kx < P; ++kx) {
          for (int c = 0; c < 3; ++c) {
            int y = py * P + ky;
            int x = px * P + kx;
            dst[(ky * P + kx) * 3 + c] = image_data[(y * W + x) * 3 + c];
          }
        }
      }
    }
  }

  // Deliver patches through the graph and collect the result synchronously.
  pimpl_->input_node->set_tensor("patches_flat", patches);
  pimpl_->input_node->set_frame_number(1);

  std::promise<std::unordered_map<std::string, dynamic_tensor>> promise;
  auto future = promise.get_future();
  pimpl_->output_node->set_callback(
      [&promise](const std::unordered_map<std::string, dynamic_tensor>& result) {
        promise.set_value(result);
      });

  pimpl_->proc->run();
  pimpl_->input_node->push();
  auto outputs = future.get();
  pimpl_->proc->stop();

  auto it = outputs.find("vision_output");
  if (it == outputs.end()) {
    throw std::runtime_error("vision_encoder: output tensor 'vision_output' not found");
  }
  return it->second;
}

}  // namespace coalsack
