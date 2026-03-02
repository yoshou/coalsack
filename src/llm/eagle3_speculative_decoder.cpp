#include "coalsack/llm/eagle3_speculative_decoder.h"

#include <spdlog/spdlog.h>

#include <algorithm>
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
#include "coalsack/llm/graph_builder.h"
#include "coalsack/nn/model_io_nodes.h"
#include "coalsack/nn/nn_nodes.h"
#include "coalsack/nn/nn_ops/add_node.h"
#include "coalsack/nn/nn_ops/concat_node.h"
#include "coalsack/nn/nn_ops/dense_ffn_node.h"
#include "coalsack/nn/nn_ops/grouped_attention_node.h"
#include "coalsack/nn/nn_ops/matmul_transpose_mixed_node.h"
#include "coalsack/nn/nn_ops/rmsnorm_node.h"
#include "coalsack/nn/result_message_nodes.h"

namespace coalsack {

namespace {

static dynamic_tensor load_gguf_tensor(const gguf_loader& loader, const std::string& name,
                                       std::ifstream& file) {
  auto info_opt = loader.get_tensor_info(name);
  if (!info_opt) {
    throw std::runtime_error("eagle3: tensor not found: " + name);
  }
  const auto& info = *info_opt;

  std::vector<int64_t> shape;
  for (int i = static_cast<int>(info.shape.size()) - 1; i >= 0; --i) {
    shape.push_back(static_cast<int64_t>(info.shape[i]));
  }
  while (shape.size() > 1 && shape.back() == 1) {
    shape.pop_back();
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
  } else if (info.type == ggml_type::I64) {
    dynamic_tensor t(dtype::int64, shape);
    std::memcpy(t.data_ptr<int64_t>(), raw.data(), info.size);
    return t;
  } else {
    dynamic_tensor t(dtype::float32, shape);
    if (!dequantize_tensor(raw.data(), t.data_ptr<float>(), numel, info.type)) {
      throw std::runtime_error("eagle3: unsupported quant type for: " + name);
    }
    return t;
  }
}

class eagle3_dec_builder {
 public:
  eagle3_dec_builder(std::shared_ptr<subgraph> graph, std::shared_ptr<graph_builder> gb)
      : graph_(std::move(graph)), gb_(std::move(gb)) {}

  graph_builder& gb() { return *gb_; }

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

 private:
  std::shared_ptr<subgraph> graph_;
  std::shared_ptr<graph_builder> gb_;
};

}  // namespace

struct eagle3_speculative_decoder::impl {
  // Config
  config cfg;
  bool loaded = false;

  // Model metadata (from GGUF)
  std::vector<int> extract_layers;  // target layers to concatenate for encoder input
  int64_t target_hidden_size = 0;   // hidden size of the target model
  int64_t hidden_size = 0;          // eagle3 hidden size (after fc projection)
  int64_t num_q_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t ffn_dim = 0;
  float rope_freq_base = 500000.0f;
  float norm_eps = 1e-5f;
  int64_t draft_vocab_size = 0;
  int64_t context_length = 131072;

  // Weights
  std::unordered_map<std::string, dynamic_tensor> weights;

  // Graph components
  std::shared_ptr<subgraph> graph;
  std::unique_ptr<graph_proc> proc;
  std::shared_ptr<model_source_node> input_node;
  std::shared_ptr<model_output_node> output_node;
  std::vector<std::shared_ptr<grouped_attention_node>> attention_nodes;

  // Per-step output state
  dynamic_tensor current_logits;
  dynamic_tensor current_prenorm;

  int64_t step_counter = 0;  // frame number for graph_proc sequencing

  // Stateless nodes used by encode() outside the graph
  matmul_transpose_mixed_node mm_node;  // fc projection: [seq, n*target_H] → [seq, H]
  rmsnorm_node rms_node;

  void build_decoder_graph();
  void reset_kv_caches();
  void initialize_kv_caches();
};

eagle3_speculative_decoder::eagle3_speculative_decoder() : pimpl_(std::make_unique<impl>()) {
  pimpl_->rms_node.set_epsilon(1e-5f);
}

eagle3_speculative_decoder::eagle3_speculative_decoder(const config& cfg)
    : pimpl_(std::make_unique<impl>()) {
  pimpl_->cfg = cfg;
  pimpl_->rms_node.set_epsilon(1e-5f);
}

eagle3_speculative_decoder::~eagle3_speculative_decoder() = default;

bool eagle3_speculative_decoder::load(const std::string& gguf_path) {
  gguf_loader loader;
  if (!loader.load(gguf_path)) {
    spdlog::error("eagle3: failed to load GGUF: {}", gguf_path);
    return false;
  }

  auto& p = *pimpl_;

  {
    auto layers = loader.get_array_int32("eagle3.extract_layers");
    if (layers.empty()) {
      spdlog::error("eagle3: missing eagle3.extract_layers");
      return false;
    }
    for (auto x : layers) p.extract_layers.push_back(static_cast<int>(x));
  }

  auto get_u32 = [&](const std::string& key) -> int64_t {
    if (auto v = loader.get_uint32(key)) return static_cast<int64_t>(*v);
    throw std::runtime_error("eagle3: missing metadata key: " + key);
  };

  p.target_hidden_size = get_u32("eagle3.target_hidden_size");
  p.hidden_size = get_u32("eagle3.embedding_length");
  p.num_q_heads = get_u32("eagle3.attention.head_count");
  p.num_kv_heads = get_u32("eagle3.attention.head_count_kv");
  p.ffn_dim = get_u32("eagle3.feed_forward_length");
  p.context_length = get_u32("eagle3.context_length");

  if (auto v = loader.get_uint32("eagle3.attention.key_length")) {
    p.head_dim = static_cast<int64_t>(*v);
  } else {
    p.head_dim = p.hidden_size / p.num_q_heads;
  }

  if (auto v = loader.get_float32("eagle3.rope.freq_base")) {
    p.rope_freq_base = *v;
  }

  if (auto v = loader.get_float32("eagle3.attention.layer_norm_rms_epsilon")) {
    p.norm_eps = *v;
  }

  spdlog::info(
      "eagle3: extract_layers=[{},...], target_H={}, H={}, n_q_heads={}, "
      "n_kv_heads={}, head_dim={}, ffn_dim={}, rope_freq_base={}",
      p.extract_layers.empty() ? -1 : p.extract_layers[0], p.target_hidden_size, p.hidden_size,
      p.num_q_heads, p.num_kv_heads, p.head_dim, p.ffn_dim, p.rope_freq_base);

  std::ifstream file(gguf_path, std::ios::binary);
  if (!file) {
    spdlog::error("eagle3: cannot open file: {}", gguf_path);
    return false;
  }

  auto load_w = [&](const std::string& name) -> dynamic_tensor {
    return load_gguf_tensor(loader, name, file);
  };

  p.weights["fc.weight"] = load_w("fc.weight");
  p.weights["hidden_norm.weight"] = load_w("blk.0.hidden_norm.weight");
  p.weights["token_embd.weight"] = load_w("token_embd.weight");
  p.weights["attn_norm.weight"] = load_w("blk.0.attn_norm.weight");
  p.weights["attn_q.weight"] = load_w("blk.0.attn_q.weight");
  p.weights["attn_k.weight"] = load_w("blk.0.attn_k.weight");
  p.weights["attn_v.weight"] = load_w("blk.0.attn_v.weight");
  p.weights["attn_output.weight"] = load_w("blk.0.attn_output.weight");
  p.weights["ffn_norm.weight"] = load_w("blk.0.ffn_norm.weight");
  p.weights["ffn_gate.weight"] = load_w("blk.0.ffn_gate.weight");
  p.weights["ffn_up.weight"] = load_w("blk.0.ffn_up.weight");
  p.weights["ffn_down.weight"] = load_w("blk.0.ffn_down.weight");
  p.weights["output_norm.weight"] = load_w("output_norm.weight");
  p.weights["output.weight"] = load_w("output.weight");
  p.weights["d2t"] = load_w("d2t");

  p.draft_vocab_size = p.weights["output.weight"].dim(0);
  spdlog::info("eagle3: draft_vocab_size={}", p.draft_vocab_size);

  p.rms_node.set_epsilon(p.norm_eps);

  p.build_decoder_graph();

  p.loaded = true;
  spdlog::info("eagle3: loaded successfully");
  return true;
}

void eagle3_speculative_decoder::impl::build_decoder_graph() {
  graph = std::make_shared<subgraph>();
  input_node = std::make_shared<model_source_node>();
  output_node = std::make_shared<model_output_node>();

  // Preload weights into the input node so they are available as graph edges
  auto set_w = [&](const std::string& name) {
    auto it = weights.find(name);
    if (it != weights.end()) {
      input_node->set_tensor(name, it->second);
    }
  };
  set_w("token_embd.weight");
  set_w("hidden_norm.weight");
  set_w("attn_norm.weight");
  set_w("attn_q.weight");
  set_w("attn_k.weight");
  set_w("attn_v.weight");
  set_w("attn_output.weight");
  set_w("ffn_norm.weight");
  set_w("ffn_gate.weight");
  set_w("ffn_up.weight");
  set_w("ffn_down.weight");
  set_w("output_norm.weight");
  set_w("output.weight");

  // Split the combined input message into individual named edges
  auto extractor = std::make_shared<result_field_extractor_node>();
  graph->add_node(extractor);
  extractor->set_input(input_node->get_output(), "default");

  auto e = [&](const std::string& name) { return extractor->add_output(name); };

  auto input_ids_e = e("input_ids");
  auto g_embd_e = e("g_embd");  // pre-computed g_embd from encode()
  auto pos_ids_e = e("position_ids");

  auto tok_embd_e = e("token_embd.weight");
  auto hidden_norm_e = e("hidden_norm.weight");
  auto attn_norm_e = e("attn_norm.weight");
  auto wq_e = e("attn_q.weight");
  auto wk_e = e("attn_k.weight");
  auto wv_e = e("attn_v.weight");
  auto wo_e = e("attn_output.weight");
  auto ffn_norm_e = e("ffn_norm.weight");
  auto ffn_gate_e = e("ffn_gate.weight");
  auto ffn_up_e = e("ffn_up.weight");
  auto ffn_down_e = e("ffn_down.weight");
  auto out_norm_e = e("output_norm.weight");
  auto out_w_e = e("output.weight");

  graph_builder::config gb_cfg;
  gb_cfg.max_seq_len = std::min(context_length, cfg.max_seq_len);
  gb_cfg.rope_freq_base = rope_freq_base;
  gb_cfg.rope_scaling_factor = 1.0f;
  gb_cfg.rope_scaling_type = "none";
  gb_cfg.use_norm_rope = true;
  auto gb = std::make_shared<graph_builder>(graph, gb_cfg);
  eagle3_dec_builder dec_gb(graph, gb);

  auto sched_e = gb->make_layer_scheduler(input_ids_e);

  auto embd_e = gb->make_embedding(sched_e, "input_ids", tok_embd_e, "token_embd.weight",
                                   "tok_embd", "embedding_lookup");

  auto embd_norm_e = gb->make_rmsnorm(embd_e, "tok_embd", attn_norm_e, "attn_norm.weight",
                                      "embd_norm", "attn_norm");

  auto g_norm_out_e = gb->make_rmsnorm(g_embd_e, "g_embd", hidden_norm_e, "hidden_norm.weight",
                                       "g_norm", "hidden_norm");

  // Concat token embedding and g_norm along the feature dim → transformer input
  auto cur_e = dec_gb.make_concat(embd_norm_e, "embd_norm", g_norm_out_e, "g_norm", 2, "cur_concat",
                                  "concat_embd_gnorm");

  auto q_proj_e = gb->make_matmul(cur_e, "cur_concat", wq_e, "attn_q.weight", "q_proj", "q_proj");
  auto k_proj_e = gb->make_matmul(cur_e, "cur_concat", wk_e, "attn_k.weight", "k_proj", "k_proj");
  auto v_proj_e = gb->make_matmul(cur_e, "cur_concat", wv_e, "attn_v.weight", "v_proj", "v_proj");

  // Reshape tensors for multi-head RoPE: [batch, seq, heads*head_dim] → [batch, seq, heads,
  // head_dim]
  dynamic_tensor q_reshape_shape(dtype::int64, {4});
  {
    auto d = q_reshape_shape.data_ptr<int64_t>();
    d[0] = 0;
    d[1] = 0;
    d[2] = num_q_heads;
    d[3] = -1;
  }
  auto q_reshape_shape_e = gb->make_shape_const(q_reshape_shape, "q_reshape_shape", q_proj_e);

  dynamic_tensor k_reshape_shape(dtype::int64, {4});
  {
    auto d = k_reshape_shape.data_ptr<int64_t>();
    d[0] = 0;
    d[1] = 0;
    d[2] = num_kv_heads;
    d[3] = -1;
  }
  auto k_reshape_shape_e = gb->make_shape_const(k_reshape_shape, "k_reshape_shape", k_proj_e);

  dynamic_tensor reshape_back_shape(dtype::int64, {3});
  {
    auto d = reshape_back_shape.data_ptr<int64_t>();
    d[0] = 0;
    d[1] = 0;
    d[2] = -1;
  }
  auto q_reshape_back_shape_e =
      gb->make_shape_const(reshape_back_shape, "q_reshape_back_shape", q_proj_e);
  auto k_reshape_back_shape_e =
      gb->make_shape_const(reshape_back_shape, "k_reshape_back_shape", k_proj_e);

  auto q_4d_e = gb->make_reshape(q_proj_e, "q_proj", q_reshape_shape_e, "q_reshape_shape", "q_4d",
                                 "q_reshape_4d");
  auto q_t_e = gb->make_transpose(q_4d_e, "q_4d", {0, 2, 1, 3}, "q_transposed", "q_transpose");
  auto q_rope_e = gb->make_rope(q_t_e, "q_transposed", pos_ids_e, nullptr, head_dim,
                                "q_rope_transposed", "rope_q");
  auto q_tb_e = gb->make_transpose(q_rope_e, "q_rope_transposed", {0, 2, 1, 3}, "q_rope_tb",
                                   "q_transpose_back");
  auto q_out_e = gb->make_reshape(q_tb_e, "q_rope_tb", q_reshape_back_shape_e,
                                  "q_reshape_back_shape", "q_rope_out", "q_reshape_3d");

  auto k_4d_e = gb->make_reshape(k_proj_e, "k_proj", k_reshape_shape_e, "k_reshape_shape", "k_4d",
                                 "k_reshape_4d");
  auto k_t_e = gb->make_transpose(k_4d_e, "k_4d", {0, 2, 1, 3}, "k_transposed", "k_transpose");
  auto k_rope_e = gb->make_rope(k_t_e, "k_transposed", pos_ids_e, nullptr, head_dim,
                                "k_rope_transposed", "rope_k");
  auto k_tb_e = gb->make_transpose(k_rope_e, "k_rope_transposed", {0, 2, 1, 3}, "k_rope_tb",
                                   "k_transpose_back");
  auto k_out_e = gb->make_reshape(k_tb_e, "k_rope_tb", k_reshape_back_shape_e,
                                  "k_reshape_back_shape", "k_rope_out", "k_reshape_3d");

  auto attn_out_e = gb->make_grouped_attn(q_out_e, "q_rope_out", k_out_e, "k_rope_out", v_proj_e,
                                          "v_proj", num_q_heads, num_kv_heads, head_dim, 0,
                                          std::nullopt, "attn_out", "grouped_attn");

  auto attn_proj_e = gb->make_matmul(attn_out_e, "attn_out", wo_e, "attn_output.weight",
                                     "attn_proj", "attn_output_proj");

  // Residual: attn output + g_norm (skip over the attention entirely)
  auto ffn_inp_e =
      gb->make_add(attn_proj_e, "attn_proj", g_norm_out_e, "g_norm", "ffn_inp", "ffn_inp_add");

  auto ffn_norm_out_e = gb->make_rmsnorm(ffn_inp_e, "ffn_inp", ffn_norm_e, "ffn_norm.weight",
                                         "ffn_norm_out", "ffn_norm");

  auto ffn_out_e =
      gb->make_dense_ffn(ffn_norm_out_e, "ffn_norm_out", ffn_gate_e, "ffn_gate.weight", ffn_up_e,
                         "ffn_up.weight", ffn_down_e, "ffn_down.weight", "ffn_out", "dense_ffn");

  // prenorm = ffn_out + ffn_inp; carried to next draft step as g_embd
  auto prenorm_e =
      gb->make_add(ffn_out_e, "ffn_out", ffn_inp_e, "ffn_inp", "prenorm", "prenorm_add");

  auto out_norm_out_e = gb->make_rmsnorm(prenorm_e, "prenorm", out_norm_e, "output_norm.weight",
                                         "out_norm_out", "output_norm");

  auto logits_e = gb->make_matmul(out_norm_out_e, "out_norm_out", out_w_e, "output.weight",
                                  "logits", "lm_head");

  auto final_sync = std::make_shared<result_message_sync_node>();
  final_sync->set_input(logits_e, "logits");
  final_sync->set_input(prenorm_e, "prenorm");
  final_sync->set_initial_ids({"logits", "prenorm"});
  graph->add_node(final_sync);

  output_node->set_input(final_sync->get_output(), "default");

  graph->add_node(input_node);
  graph->add_node(output_node);

  proc = std::make_unique<graph_proc>();
  proc->deploy(graph);

  for (uint32_t i = 0; i < graph->get_node_count(); ++i) {
    if (auto attn = std::dynamic_pointer_cast<grouped_attention_node>(graph->get_node(i))) {
      attention_nodes.push_back(attn);
    }
  }

  initialize_kv_caches();
  spdlog::info("eagle3: decoder graph built with {} attention node(s)", attention_nodes.size());
}

void eagle3_speculative_decoder::impl::initialize_kv_caches() {
  int64_t max_seq = std::min(context_length, cfg.max_seq_len);
  for (auto& attn : attention_nodes) {
    dynamic_tensor k_cache(dtype::float32, {1, num_kv_heads, max_seq, head_dim});
    dynamic_tensor v_cache(dtype::float32, {1, num_kv_heads, max_seq, head_dim});
    std::memset(k_cache.data_ptr<float>(), 0, k_cache.bytes());
    std::memset(v_cache.data_ptr<float>(), 0, v_cache.bytes());
    attn->set_k_cache(k_cache);
    attn->set_v_cache(v_cache);
  }
}

void eagle3_speculative_decoder::impl::reset_kv_caches() {
  for (auto& attn : attention_nodes) {
    attn->reset_cache();
  }
}

bool eagle3_speculative_decoder::is_loaded() const { return pimpl_->loaded; }

const std::vector<int>& eagle3_speculative_decoder::get_extract_layers() const {
  return pimpl_->extract_layers;
}
int64_t eagle3_speculative_decoder::get_target_hidden_size() const {
  return pimpl_->target_hidden_size;
}
int64_t eagle3_speculative_decoder::get_draft_vocab_size() const {
  return pimpl_->draft_vocab_size;
}

dynamic_tensor eagle3_speculative_decoder::encode(
    const std::unordered_map<int, std::vector<float>>& all_hidden_states, int64_t seq_len) const {
  auto& p = *pimpl_;

  for (int layer_idx : p.extract_layers) {
    if (all_hidden_states.find(layer_idx) == all_hidden_states.end()) {
      throw std::runtime_error("eagle3::encode: missing hidden state for layer " +
                               std::to_string(layer_idx));
    }
    auto& v = all_hidden_states.at(layer_idx);
    if (static_cast<int64_t>(v.size()) != seq_len * p.target_hidden_size) {
      throw std::runtime_error("eagle3::encode: hidden state size mismatch for layer " +
                               std::to_string(layer_idx) + " expected " +
                               std::to_string(seq_len * p.target_hidden_size) + " got " +
                               std::to_string(v.size()));
    }
  }

  int n_layers = static_cast<int>(p.extract_layers.size());
  int64_t feat_dim = static_cast<int64_t>(n_layers) * p.target_hidden_size;

  // Interleave hidden states: concat along feature dim per token → [seq_len, n_layers*target_H]
  dynamic_tensor concat_feats(dtype::float32, {seq_len, feat_dim});
  float* cf_ptr = concat_feats.data_ptr<float>();

  for (int64_t t = 0; t < seq_len; ++t) {
    for (int li = 0; li < n_layers; ++li) {
      int layer_idx = p.extract_layers[li];
      const float* src = all_hidden_states.at(layer_idx).data() + t * p.target_hidden_size;
      float* dst = cf_ptr + t * feat_dim + li * p.target_hidden_size;
      std::memcpy(dst, src, p.target_hidden_size * sizeof(float));
    }
  }

  // fc projection: [seq_len, n_layers*target_H] × fc.weight → [seq_len, eagle3_H]
  const auto& fc_weight = p.weights.at("fc.weight");
  dynamic_tensor g_embd_2d = p.mm_node.compute_test(concat_feats, fc_weight);

  return g_embd_2d.reshape({1, seq_len, p.hidden_size});
}

void eagle3_speculative_decoder::start() {
  if (!pimpl_->loaded) throw std::runtime_error("eagle3 not loaded");
  pimpl_->reset_kv_caches();
  pimpl_->step_counter = 0;
  pimpl_->proc->run();
}

void eagle3_speculative_decoder::stop() { pimpl_->proc->stop(); }

void eagle3_speculative_decoder::decode(const std::vector<uint32_t>& tokens,
                                        const dynamic_tensor& g_embd, int64_t start_pos) {
  if (!pimpl_->loaded) throw std::runtime_error("eagle3 not loaded");
  auto& p = *pimpl_;

  int64_t n = static_cast<int64_t>(tokens.size());

  if (g_embd.ndim() != 3 || g_embd.dim(0) != 1 || g_embd.dim(1) != n ||
      g_embd.dim(2) != p.hidden_size) {
    throw std::runtime_error("eagle3::decode: g_embd shape mismatch; expected [1, " +
                             std::to_string(n) + ", " + std::to_string(p.hidden_size) + "] got [" +
                             std::to_string(g_embd.dim(0)) + ", " + std::to_string(g_embd.dim(1)) +
                             ", " + std::to_string(g_embd.dim(2)) + "]");
  }

  // Build input tensors for this decode step
  dynamic_tensor input_ids_t(dtype::int32, {1, n});
  {
    int32_t* ids = input_ids_t.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i) {
      ids[i] = static_cast<int32_t>(tokens[i]);
    }
  }

  dynamic_tensor pos_ids_t(dtype::int64, {n});  // absolute positions for RoPE
  {
    int64_t* pos = pos_ids_t.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      pos[i] = start_pos + i;
    }
  }

  p.input_node->set_tensor("input_ids", input_ids_t);
  p.input_node->set_tensor("g_embd", g_embd);
  p.input_node->set_tensor("position_ids", pos_ids_t);
  p.input_node->set_frame_number(++p.step_counter);

  // Push input and block until graph finishes
  std::promise<std::unordered_map<std::string, dynamic_tensor>> promise;
  auto future = promise.get_future();
  p.output_node->set_callback(
      [&promise](const std::unordered_map<std::string, dynamic_tensor>& result) {
        promise.set_value(result);
      });

  p.input_node->push();
  auto outputs = future.get();

  auto get_output = [&](const std::string& key) -> dynamic_tensor {
    auto it = outputs.find(key);
    if (it == outputs.end()) {
      throw std::runtime_error("eagle3::decode: output '" + key + "' not found");
    }
    return it->second;
  };

  p.current_logits = get_output("logits");
  p.current_prenorm = get_output("prenorm");
}

const dynamic_tensor& eagle3_speculative_decoder::get_logits() const {
  return pimpl_->current_logits;
}

const dynamic_tensor& eagle3_speculative_decoder::get_prenorm() const {
  return pimpl_->current_prenorm;
}

int64_t eagle3_speculative_decoder::draft_to_target(int64_t draft_id) const {
  const auto& d2t = pimpl_->weights.at("d2t");
  if (draft_id < 0 || draft_id >= d2t.dim(0)) {
    throw std::runtime_error("eagle3::draft_to_target: draft_id " + std::to_string(draft_id) +
                             " out of range");
  }
  return draft_id + d2t.data_ptr<int64_t>()[draft_id];
}

}  // namespace coalsack
