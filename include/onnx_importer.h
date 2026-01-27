#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_proc.h"
#include "nn_nodes.h"

// Forward declare ONNX types
namespace onnx {
class ModelProto;
class GraphProto;
class NodeProto;
class TensorProto;
class ValueInfoProto;
}  // namespace onnx

namespace coalsack {

class model_input_node;
class model_output_node;

class onnx_importer {
 private:
  std::shared_ptr<subgraph> graph_;
  std::unordered_map<std::string, graph_edge_ptr> value_edges_;
  std::unordered_map<std::string, graph_node_ptr> value_producers_;
  std::unordered_map<std::string, dynamic_tensor> initializers_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::shared_ptr<constant_node>> constants_;

 public:
  onnx_importer() : graph_(std::make_shared<subgraph>()) {}

  bool load_model(const std::string& filepath, int max_nodes = -1);

  std::shared_ptr<subgraph> get_subgraph() const { return graph_; }
  const std::unordered_map<std::string, dynamic_tensor>& get_initializers() const {
    return initializers_;
  }
  const std::vector<std::string>& get_input_names() const { return input_names_; }
  const std::vector<std::string>& get_output_names() const { return output_names_; }

  graph_edge_ptr get_value_edge(const std::string& name) const {
    auto it = value_edges_.find(name);
    if (it != value_edges_.end()) {
      return it->second;
    }
    return nullptr;
  }

  bool wire_io_nodes(std::shared_ptr<model_input_node> input_node,
                     std::shared_ptr<model_output_node> output_node);

 private:
  bool import_graph(const onnx::GraphProto& graph, int max_nodes = -1);
  bool import_node(const onnx::NodeProto& node);
  bool import_initializers(const onnx::GraphProto& graph);
  dynamic_tensor convert_tensor_proto(const onnx::TensorProto& tensor);
  void parse_attributes(const onnx::NodeProto& node, graph_node_ptr coalsack_node);
  graph_node_ptr create_node_for_op(const std::string& op_type);
  void set_node_name(const onnx::NodeProto& node, graph_node_ptr coalsack_node);
  void collect_input_edges(const onnx::NodeProto& node, std::vector<std::string>& input_names,
                           std::vector<graph_edge_ptr>& input_edges);
  void setup_input_connections(const onnx::NodeProto& node, graph_node_ptr coalsack_node,
                               const std::vector<std::string>& input_names,
                               const std::vector<graph_edge_ptr>& input_edges);
  void setup_output_connections(const onnx::NodeProto& node, graph_node_ptr coalsack_node,
                                const std::string& op_type);
};

inline graph_node_ptr onnx_importer::create_node_for_op(const std::string& op_type) {
  if (op_type == "Shape") return std::make_shared<shape_node>();
  if (op_type == "Reshape") return std::make_shared<reshape_node>();
  if (op_type == "Unsqueeze") return std::make_shared<unsqueeze_node>();
  if (op_type == "Squeeze") return std::make_shared<squeeze_node>();
  if (op_type == "Transpose") return std::make_shared<transpose_node>();
  if (op_type == "Cast") return std::make_shared<cast_node>();
  if (op_type == "Gather") return std::make_shared<gather_node>();
  if (op_type == "Concat") return std::make_shared<concat_node>();
  if (op_type == "Slice") return std::make_shared<slice_node>();
  if (op_type == "Constant") return std::make_shared<constant_node>();
  if (op_type == "ConstantOfShape") return std::make_shared<constant_of_shape_node>();
  if (op_type == "Expand") return std::make_shared<expand_node>();
  if (op_type == "Range") return std::make_shared<range_node>();
  if (op_type == "Flatten") return std::make_shared<flatten_node>();
  if (op_type == "Tile") return std::make_shared<tile_node>();

  if (op_type == "Add") return std::make_shared<add_node>();
  if (op_type == "Sub") return std::make_shared<sub_node>();
  if (op_type == "Mul") return std::make_shared<mul_node>();
  if (op_type == "Div") return std::make_shared<div_node>();
  if (op_type == "Sqrt") return std::make_shared<sqrt_node>();
  if (op_type == "Pow") return std::make_shared<pow_node>();
  if (op_type == "Relu") return std::make_shared<relu_node>();
  if (op_type == "Exp") return std::make_shared<exp_node>();
  if (op_type == "Log") return std::make_shared<log_node>();
  if (op_type == "Sigmoid") return std::make_shared<sigmoid_node>();
  if (op_type == "Tanh") return std::make_shared<tanh_node>();
  if (op_type == "Neg") return std::make_shared<neg_node>();
  if (op_type == "Erf") return std::make_shared<erf_node>();
  if (op_type == "Cos") return std::make_shared<cos_node>();
  if (op_type == "Sin") return std::make_shared<sin_node>();
  if (op_type == "Clip") return std::make_shared<clip_node>();
  if (op_type == "Mod") return std::make_shared<mod_node>();

  if (op_type == "Equal") return std::make_shared<equal_node>();
  if (op_type == "Greater") return std::make_shared<greater_node>();
  if (op_type == "Where") return std::make_shared<where_node>();
  if (op_type == "And") return std::make_shared<and_node>();
  if (op_type == "Or") return std::make_shared<or_node>();
  if (op_type == "Not") return std::make_shared<not_node>();

  if (op_type == "MatMul") return std::make_shared<matmul_node>();
  if (op_type == "Softmax") return std::make_shared<softmax_node>();
  if (op_type == "LogSoftmax") return std::make_shared<log_softmax_node>();
  if (op_type == "Conv") return std::make_shared<conv_node>();
  if (op_type == "MaxPool") return std::make_shared<max_pool_node>();
  if (op_type == "LayerNormalization") return std::make_shared<layer_normalization_node>();
  if (op_type == "ReduceMean") return std::make_shared<reduce_mean_node>();
  if (op_type == "ReduceL2") return std::make_shared<reduce_l2_node>();

  if (op_type == "GatherElements") return std::make_shared<gather_elements_node>();
  if (op_type == "TopK") return std::make_shared<topk_node>();
  if (op_type == "NonZero") return std::make_shared<non_zero_node>();
  if (op_type == "Split") return std::make_shared<split_node>();
  if (op_type == "ScatterND") return std::make_shared<scatter_nd_node>();
  if (op_type == "GridSample") return std::make_shared<grid_sample_node>();

  throw std::runtime_error("ONNX op not implemented: " + op_type);
}

}  // namespace coalsack
