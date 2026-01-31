#include "gpt_oss_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

#include <spdlog/spdlog.h>

#include "gguf_dequant.h"
#include "gguf_loader.h"
#include "gpt2_tokenizer.h"
#include "graph_proc.h"
#include "model_io_nodes.h"
#include "nn_nodes.h"
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

  // Compute head_dim
  if (pimpl_->num_q_heads > 0) {
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

    // Create output tensor
    dynamic_tensor tensor(dtype::float32, shape);
    float* data = tensor.data_ptr<float>();

    // Dequantize or copy data
    if (info.type == ggml_type::F32) {
      // Direct copy for float32
      std::memcpy(data, raw_data.data(), info.size);
      ++loaded_count;
    } else if (dequantize_tensor(raw_data.data(), data, numel, info.type)) {
      ++dequantized_count;
    } else {
      std::cerr << "WARNING: Unsupported quantization type for: " << name << "\n";
      ++skipped_count;
      continue;
    }

    pimpl_->weights[name] = std::move(tensor);
  }
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION during weight loading at tensor " << tensor_idx << ": " << e.what() << "\n";
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
  // Start with minimal test graph for validation
  // Later can switch to full 24-layer graph
  build_minimal_test_graph();
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
  auto output_proj = std::make_shared<matmul_node>();

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

void gpt_oss_engine::wire_io_nodes(graph_edge_ptr input_placeholder,
                                    graph_edge_ptr logits_output) {
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

uint32_t gpt_oss_engine::sample_token(const float* logits, int64_t vocab_size,
                                      float temperature) {
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
