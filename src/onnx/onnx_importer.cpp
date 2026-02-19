#include "coalsack/onnx/onnx_importer.h"

#include <cstdlib>
#include <iostream>

#include "coalsack/nn/model_io_nodes.h"
#include "coalsack/nn/result_message_nodes.h"

namespace coalsack {

bool onnx_importer::load_model(const std::string& filepath, int max_nodes) {
  onnx_model_proto model;
  if (!load_onnx_model(filepath, model)) {
    std::cerr << "Failed to parse ONNX model from file: " << filepath << "\n";
    return false;
  }

  if (!model.has_graph()) {
    std::cerr << "ONNX model has no graph\n";
    return false;
  }

  return import_graph(*model.graph, max_nodes);
}

bool onnx_importer::import_graph(const onnx_graph_proto& graph, int max_nodes) {
  if (!import_initializers(graph)) {
    std::cerr << "Failed to import initializers\n";
    return false;
  }

  for (const auto& input : graph.input) {
    input_names_.push_back(input.name);
  }

  for (const auto& output : graph.output) {
    output_names_.push_back(output.name);
  }

  std::unordered_map<std::string, int> value_producer;
  int node_count = static_cast<int>(graph.node.size());
  std::vector<std::vector<int>> adj(node_count);
  std::vector<int> in_degree(node_count, 0);

  for (int i = 0; i < node_count; ++i) {
    const auto& node = graph.node[i];
    for (const auto& out : node.output) {
      value_producer[out] = i;
    }
  }

  for (int i = 0; i < node_count; ++i) {
    const auto& node = graph.node[i];
    for (const auto& input : node.input) {
      auto it = value_producer.find(input);
      if (it != value_producer.end()) {
        int producer_idx = it->second;
        adj[producer_idx].push_back(i);
        in_degree[i]++;
      }
    }
  }

  std::vector<int> sorted_nodes;
  std::vector<int> queue;

  for (int i = 0; i < node_count; ++i) {
    if (in_degree[i] == 0) {
      queue.push_back(i);
    }
  }

  int processed_count = 0;
  while (processed_count < static_cast<int>(queue.size())) {
    int u = queue[processed_count++];
    sorted_nodes.push_back(u);

    for (int v : adj[u]) {
      in_degree[v]--;
      if (in_degree[v] == 0) {
        queue.push_back(v);
      }
    }
  }

  if (static_cast<int>(sorted_nodes.size()) < node_count) {
    std::cerr << "Cycle detected in ONNX graph! Sorted " << sorted_nodes.size() << " of "
              << node_count << " nodes.\n";
    std::cerr << "The following nodes are part of (or reachable from) the cycle:\n";
    int cycle_node_count = 0;
    for (int i = 0; i < node_count; ++i) {
      if (in_degree[i] > 0) {
        std::cerr << "  Cycle Node: Index=" << i << " Op=" << graph.node[i].op_type
                  << " Name=" << graph.node[i].name << "\n";
        cycle_node_count++;
        if (cycle_node_count >= 10) {
          std::cerr << "  ... and more\n";
          break;
        }
      }
    }
    return false;
  }

  int count = static_cast<int>(sorted_nodes.size());
  if (max_nodes > 0 && max_nodes < count) {
    count = max_nodes;
    std::cout << "Importing partial graph: " << count << "/" << node_count << " nodes\n";
  }

  for (int i = 0; i < count; ++i) {
    int node_idx = sorted_nodes[i];
    if (!import_node(graph.node[node_idx])) {
      std::cerr << "Failed to import node " << node_idx << " (" << graph.node[node_idx].op_type
                << ")\n";
      return false;
    }
  }

  std::cerr << "Successfully imported " << count << " nodes\n";
  return true;
}

bool onnx_importer::import_initializers(const onnx_graph_proto& graph) {
  for (const auto& tensor_proto : graph.initializer) {
    const std::string& name = tensor_proto.name;

    try {
      dynamic_tensor tensor = convert_tensor_proto(tensor_proto);
      initializers_[name] = tensor;
    } catch (const std::exception& e) {
      return false;
    }
  }
  return true;
}

dynamic_tensor onnx_importer::convert_tensor_proto(const onnx_tensor_proto& tensor_proto) {
  std::vector<int64_t> shape;
  for (int64_t d : tensor_proto.dims) {
    shape.push_back(d);
  }

  dtype dt;
  switch (tensor_proto.data_type) {
    case onnx_data_type::float32:
      dt = dtype::float32;
      break;
    case onnx_data_type::int64:
      dt = dtype::int64;
      break;
    case onnx_data_type::int32:
      dt = dtype::int32;
      break;
    case onnx_data_type::bool_:
      dt = dtype::bool_;
      break;
    case onnx_data_type::float64:
      dt = dtype::float64;
      break;
    default:
      throw std::runtime_error("Unsupported tensor dtype");
  }

  dynamic_tensor tensor(dt, shape);

  int64_t numel = 1;
  for (auto dim : shape) {
    numel *= dim;
  }

  if (numel <= 0) {
    return tensor;
  }

  if (dt == dtype::float32) {
    float* data = tensor.data_ptr<float>();
    if (!data) {
      return tensor;
    }
    if (!tensor_proto.float_data.empty()) {
      int count = std::min<int>(numel, static_cast<int>(tensor_proto.float_data.size()));
      for (int i = 0; i < count; ++i) {
        data[i] = tensor_proto.float_data[i];
      }
    } else if (!tensor_proto.raw_data.empty()) {
      const std::string& raw = tensor_proto.raw_data;
      size_t copy_size = std::min<size_t>(raw.size(), numel * sizeof(float));
      if (copy_size > 0) {
        std::memcpy(data, raw.data(), copy_size);
      }
    }
  } else if (dt == dtype::int64) {
    int64_t* data = tensor.data_ptr<int64_t>();
    if (!data) {
      return tensor;
    }
    if (!tensor_proto.int64_data.empty()) {
      int count = std::min<int>(numel, static_cast<int>(tensor_proto.int64_data.size()));
      for (int i = 0; i < count; ++i) {
        data[i] = tensor_proto.int64_data[i];
      }
    } else if (!tensor_proto.raw_data.empty()) {
      const std::string& raw = tensor_proto.raw_data;
      size_t copy_size = std::min<size_t>(raw.size(), numel * sizeof(int64_t));
      if (copy_size > 0) {
        std::memcpy(data, raw.data(), copy_size);
      }
    }
  } else if (dt == dtype::int32) {
    int32_t* data = tensor.data_ptr<int32_t>();
    if (!data) {
      return tensor;
    }
    if (!tensor_proto.int32_data.empty()) {
      int count = std::min<int>(numel, static_cast<int>(tensor_proto.int32_data.size()));
      for (int i = 0; i < count; ++i) {
        data[i] = tensor_proto.int32_data[i];
      }
    } else if (!tensor_proto.raw_data.empty()) {
      const std::string& raw = tensor_proto.raw_data;
      size_t copy_size = std::min<size_t>(raw.size(), numel * sizeof(int32_t));
      if (copy_size > 0) {
        std::memcpy(data, raw.data(), copy_size);
      }
    }
  }
  return tensor;
}

void onnx_importer::set_node_name(const onnx_node_proto& node, graph_node_ptr coalsack_node) {
  if (auto unary_node = std::dynamic_pointer_cast<unary_op_node>(coalsack_node)) {
    unary_node->set_node_name(node.name);
  } else if (auto binary_node = std::dynamic_pointer_cast<binary_op_node>(coalsack_node)) {
    binary_node->set_node_name(node.name);
  } else if (auto variadic_node = std::dynamic_pointer_cast<variadic_op_node>(coalsack_node)) {
    variadic_node->set_node_name(node.name);
  } else if (auto gather = std::dynamic_pointer_cast<gather_node>(coalsack_node)) {
    gather->set_node_name(node.name);
  } else if (auto unsqueeze = std::dynamic_pointer_cast<unsqueeze_node>(coalsack_node)) {
    unsqueeze->set_node_name(node.name);
  } else if (auto slice = std::dynamic_pointer_cast<slice_node>(coalsack_node)) {
    slice->set_node_name(node.name);
  } else if (auto reshape = std::dynamic_pointer_cast<reshape_node>(coalsack_node)) {
    reshape->set_node_name(node.name);
  } else if (auto squeeze = std::dynamic_pointer_cast<squeeze_node>(coalsack_node)) {
    squeeze->set_node_name(node.name);
  }
}

void onnx_importer::collect_input_edges(const onnx_node_proto& node,
                                        std::vector<std::string>& input_names,
                                        std::vector<graph_edge_ptr>& input_edges) {
  for (const auto& input_name : node.input) {
    if (input_name.empty()) continue;

    graph_edge_ptr input_edge;
    auto init_it = initializers_.find(input_name);
    if (init_it != initializers_.end()) {
      auto const_node = std::make_shared<constant_node>(init_it->second, input_name);
      constants_.push_back(const_node);
      graph_->add_node(const_node);
      input_edge = const_node->get_output("default");
      value_edges_[input_name] = input_edge;
      value_producers_[input_name] = const_node;
    } else {
      auto it = value_edges_.find(input_name);
      if (it != value_edges_.end()) {
        input_edge = it->second;
      } else {
        input_edge = std::make_shared<graph_edge>(nullptr);
        value_edges_[input_name] = input_edge;
      }
    }

    input_names.push_back(input_name);
    input_edges.push_back(input_edge);
  }
}

void onnx_importer::setup_input_connections(const onnx_node_proto& node,
                                            graph_node_ptr coalsack_node,
                                            const std::vector<std::string>& all_input_names,
                                            const std::vector<graph_edge_ptr>& all_input_edges) {
  if (all_input_names.size() >= 2) {
    auto sync_node = std::make_shared<result_message_sync_node>();

    for (size_t i = 0; i < all_input_names.size(); ++i) {
      sync_node->set_input(all_input_edges[i], all_input_names[i]);
    }
    sync_node->set_initial_ids(all_input_names);
    graph_->add_node(sync_node);

    auto sync_output = sync_node->get_output();
    coalsack_node->set_input(sync_output, "default");
    if (auto binary_node = std::dynamic_pointer_cast<binary_op_node>(coalsack_node)) {
      if (all_input_names.size() == 2) {
        binary_node->set_input_names(all_input_names[0], all_input_names[1]);
      }
    } else if (auto variadic_node = std::dynamic_pointer_cast<variadic_op_node>(coalsack_node)) {
      variadic_node->set_input_names(all_input_names);
    } else if (auto unsqueeze_node_ptr = std::dynamic_pointer_cast<unsqueeze_node>(coalsack_node)) {
      if (all_input_names.size() >= 1) {
        std::string axes_name = (all_input_names.size() > 1) ? all_input_names[1] : "";
        unsqueeze_node_ptr->set_input_names(all_input_names[0], axes_name);
      }
    } else if (auto squeeze_node_ptr = std::dynamic_pointer_cast<squeeze_node>(coalsack_node)) {
      if (all_input_names.size() >= 1) {
        std::string axes_name = (all_input_names.size() > 1) ? all_input_names[1] : "";
        squeeze_node_ptr->set_input_names(all_input_names[0], axes_name);
      }
    } else if (auto gather_node_ptr = std::dynamic_pointer_cast<gather_node>(coalsack_node)) {
      if (all_input_names.size() == 2) {
        gather_node_ptr->set_input_names(all_input_names[0], all_input_names[1]);
      }
    } else if (auto reshape_node_ptr = std::dynamic_pointer_cast<reshape_node>(coalsack_node)) {
      if (all_input_names.size() == 2) {
        reshape_node_ptr->set_input_names(all_input_names[0], all_input_names[1]);
      }
    } else if (auto slice_node_ptr = std::dynamic_pointer_cast<slice_node>(coalsack_node)) {
      if (all_input_names.size() >= 3) {
        std::string axes_name = (all_input_names.size() > 3) ? all_input_names[3] : "";
        std::string steps_name = (all_input_names.size() > 4) ? all_input_names[4] : "";
        slice_node_ptr->set_input_names(all_input_names[0], all_input_names[1], all_input_names[2],
                                        axes_name, steps_name);
      }
    } else if (auto topk_node_ptr = std::dynamic_pointer_cast<topk_node>(coalsack_node)) {
      if (all_input_names.size() == 2) {
        topk_node_ptr->set_input_names(all_input_names[0], all_input_names[1]);
      }
    } else if (auto unary_node = std::dynamic_pointer_cast<unary_op_node>(coalsack_node)) {
      if (all_input_names.size() >= 1) {
        unary_node->set_input_name(all_input_names[0]);
      }
    }
  } else {
    for (size_t i = 0; i < all_input_names.size(); ++i) {
      std::string input_port = (all_input_names.size() == 1) ? "default" : all_input_names[i];
      coalsack_node->set_input(all_input_edges[i], input_port);
    }
    if (all_input_names.size() == 1) {
      if (auto unary_node = std::dynamic_pointer_cast<unary_op_node>(coalsack_node)) {
        unary_node->set_input_name(all_input_names[0]);
      } else if (auto variadic_node = std::dynamic_pointer_cast<variadic_op_node>(coalsack_node)) {
        variadic_node->set_input_names(all_input_names);
      }
    }
  }
}

void onnx_importer::setup_output_connections(const onnx_node_proto& node,
                                             graph_node_ptr coalsack_node,
                                             const std::string& op_type) {
  if (op_type == "TopK" && static_cast<int>(node.output.size()) >= 2) {
    auto topk = std::dynamic_pointer_cast<topk_node>(coalsack_node);
    if (topk) {
      topk->set_values_output_name(node.output[0]);
      topk->set_indices_output_name(node.output[1]);
    }
  } else if (op_type == "Split" && static_cast<int>(node.output.size()) > 1) {
    auto split = std::dynamic_pointer_cast<split_node>(coalsack_node);
    if (split) {
      std::vector<std::string> output_names;
      for (const auto& out : node.output) {
        output_names.push_back(out);
      }
      split->set_output_names(output_names);
    }
  }

  for (int i = 0; i < static_cast<int>(node.output.size()); ++i) {
    const std::string& output_name = node.output[i];

    graph_edge_ptr output_edge;
    if (static_cast<int>(node.output.size()) == 1) {
      output_edge = coalsack_node->get_output("default");

      if (auto unary_node = std::dynamic_pointer_cast<unary_op_node>(coalsack_node)) {
        unary_node->set_output_name(output_name);
      } else if (auto binary_node = std::dynamic_pointer_cast<binary_op_node>(coalsack_node)) {
        binary_node->set_output_name(output_name);
      } else if (auto variadic_node = std::dynamic_pointer_cast<variadic_op_node>(coalsack_node)) {
        variadic_node->set_output_name(output_name);
      } else if (auto squeeze = std::dynamic_pointer_cast<squeeze_node>(coalsack_node)) {
        squeeze->set_output_name(output_name);
      } else if (auto unsqueeze = std::dynamic_pointer_cast<unsqueeze_node>(coalsack_node)) {
        unsqueeze->set_output_name(output_name);
      } else if (auto reshape = std::dynamic_pointer_cast<reshape_node>(coalsack_node)) {
        reshape->set_output_name(output_name);
      } else if (auto slice = std::dynamic_pointer_cast<slice_node>(coalsack_node)) {
        slice->set_output_name(output_name);
      } else if (auto gather = std::dynamic_pointer_cast<gather_node>(coalsack_node)) {
        gather->set_output_name(output_name);
      } else if (auto split = std::dynamic_pointer_cast<split_node>(coalsack_node)) {
      }
    } else {
      if (op_type == "TopK") {
        output_edge = coalsack_node->get_output("default");
      } else if (op_type == "Split") {
        output_edge = coalsack_node->get_output("default");
      } else {
        output_edge = coalsack_node->get_output(output_name);
      }
    }

    if (!output_edge) {
      throw std::runtime_error("Output edge not found: " + output_name);
    }
    value_edges_[output_name] = output_edge;
    value_producers_[output_name] = coalsack_node;
  }
}

bool onnx_importer::import_node(const onnx_node_proto& node) {
  const std::string& op_type = node.op_type;

  try {
    if (op_type == "Constant") {
      dynamic_tensor value;
      for (const auto& attr : node.attribute) {
        if (attr.name == "value" && attr.has_t()) {
          value = convert_tensor_proto(*attr.t);
          break;
        }
      }

      std::string output_name = !node.output.empty() ? node.output[0] : "constant_output";
      auto const_node = std::make_shared<constant_node>(value, output_name);
      constants_.push_back(const_node);
      graph_->add_node(const_node);

      if (!node.output.empty()) {
        const std::string& out_name = node.output[0];
        auto output_edge = const_node->get_output("default");
        value_edges_[out_name] = output_edge;
        value_producers_[out_name] = const_node;
      }

      return true;
    }
    auto coalsack_node = create_node_for_op(op_type);
    graph_->add_node(coalsack_node);
    parse_attributes(node, coalsack_node);
    set_node_name(node, coalsack_node);

    std::vector<std::string> all_input_names;
    std::vector<graph_edge_ptr> all_input_edges;
    collect_input_edges(node, all_input_names, all_input_edges);
    setup_input_connections(node, coalsack_node, all_input_names, all_input_edges);
    setup_output_connections(node, coalsack_node, op_type);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to import node " << op_type << ": " << e.what() << "\n";
    return false;
  }
}

void onnx_importer::parse_attributes(const onnx_node_proto& node, graph_node_ptr coalsack_node) {
  const std::string& op_type = node.op_type;

  for (const auto& attr : node.attribute) {
    const std::string& attr_name = attr.name;

    if (op_type == "Conv") {
      auto conv_node_ptr = std::dynamic_pointer_cast<conv_node>(coalsack_node);
      if (conv_node_ptr) {
        if (attr_name == "strides" && !attr.ints.empty()) {
          conv_node_ptr->set_strides(attr.ints);
        } else if (attr_name == "pads" && !attr.ints.empty()) {
          conv_node_ptr->set_pads(attr.ints);
        } else if (attr_name == "dilations" && !attr.ints.empty()) {
          conv_node_ptr->set_dilations(attr.ints);
        } else if (attr_name == "group" && attr.has_i()) {
          conv_node_ptr->set_group(*attr.i);
        }
      }
    } else if (op_type == "MaxPool") {
      auto pool_node = std::dynamic_pointer_cast<max_pool_node>(coalsack_node);
      if (pool_node) {
        if (attr_name == "kernel_shape" && !attr.ints.empty()) {
          pool_node->set_kernel_shape(attr.ints);
        } else if (attr_name == "strides" && !attr.ints.empty()) {
          pool_node->set_strides(attr.ints);
        } else if (attr_name == "pads" && !attr.ints.empty()) {
          pool_node->set_pads(attr.ints);
        }
      }
    } else if (op_type == "Transpose") {
      auto trans_node = std::dynamic_pointer_cast<transpose_node>(coalsack_node);
      if (trans_node && attr_name == "perm" && !attr.ints.empty()) {
        trans_node->set_perm(attr.ints);
      }
    } else if (op_type == "Unsqueeze") {
      auto unsqueeze_node_ptr = std::dynamic_pointer_cast<unsqueeze_node>(coalsack_node);
      if (unsqueeze_node_ptr && attr_name == "axes" && !attr.ints.empty()) {
        unsqueeze_node_ptr->set_axes(attr.ints);
      }
    } else if (op_type == "Squeeze") {
      auto squeeze_node_ptr = std::dynamic_pointer_cast<squeeze_node>(coalsack_node);
      if (squeeze_node_ptr && attr_name == "axes" && !attr.ints.empty()) {
        squeeze_node_ptr->set_axes(attr.ints);
      }
    } else if (op_type == "Split") {
      auto split_node_ptr = std::dynamic_pointer_cast<split_node>(coalsack_node);
      if (split_node_ptr) {
        if (attr_name == "axis" && attr.has_i()) {
          split_node_ptr->set_axis(*attr.i);
        } else if (attr_name == "split" && !attr.ints.empty()) {
          split_node_ptr->set_splits(attr.ints);
        }
      }
    } else if (op_type == "Cast") {
      auto cast_node_ptr = std::dynamic_pointer_cast<cast_node>(coalsack_node);
      if (cast_node_ptr && attr_name == "to" && attr.has_i()) {
        cast_node_ptr->set_to_dtype(*attr.i);
      }
    } else if (op_type == "ReduceMean") {
      auto reduce_node = std::dynamic_pointer_cast<reduce_mean_node>(coalsack_node);
      if (reduce_node) {
        if (attr_name == "axes" && !attr.ints.empty()) {
          reduce_node->set_axes(attr.ints);
        } else if (attr_name == "keepdims" && attr.has_i()) {
          reduce_node->set_keepdims(*attr.i != 0);
        }
      }
    } else if (op_type == "ReduceL2") {
      auto reduce_node = std::dynamic_pointer_cast<reduce_l2_node>(coalsack_node);
      if (reduce_node) {
        if (attr_name == "axes" && !attr.ints.empty()) {
          reduce_node->set_axes(attr.ints);
        } else if (attr_name == "keepdims" && attr.has_i()) {
          reduce_node->set_keepdims(*attr.i != 0);
        }
      }
    } else if (op_type == "LayerNormalization") {
      auto ln_node = std::dynamic_pointer_cast<layer_normalization_node>(coalsack_node);
      if (ln_node) {
        if (attr_name == "axis" && attr.has_i()) {
          ln_node->set_axis(*attr.i);
        } else if (attr_name == "epsilon" && attr.has_f()) {
          ln_node->set_epsilon(*attr.f);
        }
      }
    } else if (op_type == "Softmax") {
      auto sm_node = std::dynamic_pointer_cast<softmax_node>(coalsack_node);
      if (sm_node && attr_name == "axis" && attr.has_i()) {
        sm_node->set_axis(*attr.i);
      }
    } else if (op_type == "LogSoftmax") {
      auto lsm_node = std::dynamic_pointer_cast<log_softmax_node>(coalsack_node);
      if (lsm_node && attr_name == "axis" && attr.has_i()) {
        lsm_node->set_axis(*attr.i);
      }
    } else if (op_type == "Concat") {
      auto concat_node_ptr = std::dynamic_pointer_cast<concat_node>(coalsack_node);
      if (concat_node_ptr && attr_name == "axis" && attr.has_i()) {
        concat_node_ptr->set_axis(*attr.i);
      }
    } else if (op_type == "ConstantOfShape") {
      auto cos_node = std::dynamic_pointer_cast<constant_of_shape_node>(coalsack_node);
      if (cos_node && attr_name == "value" && attr.has_t()) {
        cos_node->set_value(convert_tensor_proto(*attr.t));
      }
    } else if (op_type == "Gather") {
      auto gather_node_ptr = std::dynamic_pointer_cast<gather_node>(coalsack_node);
      if (gather_node_ptr && attr_name == "axis" && attr.has_i()) {
        gather_node_ptr->set_axis(*attr.i);
      }
    } else if (op_type == "Clip") {
      auto clip_node_ptr = std::dynamic_pointer_cast<clip_node>(coalsack_node);
      if (clip_node_ptr) {
        if (attr_name == "min" && attr.has_f()) {
          clip_node_ptr->set_min(*attr.f);
        } else if (attr_name == "max" && attr.has_f()) {
          clip_node_ptr->set_max(*attr.f);
        }
      }
    } else if (op_type == "TopK") {
      auto topk_node_ptr = std::dynamic_pointer_cast<topk_node>(coalsack_node);
      if (topk_node_ptr) {
        if (attr_name == "axis" && attr.has_i()) {
          topk_node_ptr->set_axis(*attr.i);
        } else if (attr_name == "largest" && attr.has_i()) {
          topk_node_ptr->set_largest(*attr.i != 0);
        }
      }
    } else if (op_type == "Flatten") {
      auto flatten_node_ptr = std::dynamic_pointer_cast<flatten_node>(coalsack_node);
      if (flatten_node_ptr && attr_name == "axis" && attr.has_i()) {
        flatten_node_ptr->set_axis(*attr.i);
      }
    } else if (op_type == "GatherElements") {
      auto gather_elements_node_ptr =
          std::dynamic_pointer_cast<gather_elements_node>(coalsack_node);
      if (gather_elements_node_ptr && attr_name == "axis" && attr.has_i()) {
        gather_elements_node_ptr->set_axis(*attr.i);
      }
    } else if (op_type == "Mod") {
      auto mod_node_ptr = std::dynamic_pointer_cast<mod_node>(coalsack_node);
      if (mod_node_ptr && attr_name == "fmod" && attr.has_i()) {
        mod_node_ptr->set_fmod(*attr.i);
      }
    } else if (op_type == "GridSample") {
      auto grid_node_ptr = std::dynamic_pointer_cast<grid_sample_node>(coalsack_node);
      if (grid_node_ptr) {
        if (attr_name == "align_corners" && attr.has_i()) {
          grid_node_ptr->set_align_corners(*attr.i);
        } else if (attr_name == "mode" && attr.has_s()) {
          grid_node_ptr->set_mode(*attr.s);
        } else if (attr_name == "padding_mode" && attr.has_s()) {
          grid_node_ptr->set_padding_mode(*attr.s);
        }
      }
    }
  }
}

bool onnx_importer::wire_io_nodes(std::shared_ptr<model_input_node> input_node,
                                  std::shared_ptr<model_output_node> output_node) {
  try {
    auto extractor = std::make_shared<result_field_extractor_node>();
    graph_->add_node(extractor);

    auto input_output = input_node->get_output();
    extractor->set_input(input_output, "default");

    for (const auto& input_name : input_names_) {
      auto extractor_output = extractor->add_output(input_name);
      auto graph_input_edge = get_value_edge(input_name);

      if (!graph_input_edge) {
        throw std::runtime_error("Input edge not found: " + input_name);
      }

      for (uint32_t i = 0; i < graph_->get_node_count(); ++i) {
        auto node = graph_->get_node(i);
        const auto& inputs = node->get_inputs();
        for (const auto& [port_name, input_edge] : inputs) {
          if (input_edge == graph_input_edge) {
            node->set_input(extractor_output, port_name);
          }
        }
      }

      value_edges_[input_name] = extractor_output;
    }
    if (!constants_.empty() && extractor) {
      auto frame_source = extractor->get_output(input_names_[0]);
      for (auto& const_node : constants_) {
        const_node->set_input(frame_source, "default");
      }
    }

    auto sync_node = std::make_shared<result_message_sync_node>();
    graph_->add_node(sync_node);
    std::vector<std::string> output_input_names;

    for (const auto& output_name : output_names_) {
      auto graph_output_edge = get_value_edge(output_name);
      if (!graph_output_edge) {
        throw std::runtime_error("Output edge not found: " + output_name);
      }

      sync_node->set_input(graph_output_edge, output_name);
      output_input_names.push_back(output_name);
    }
    sync_node->set_initial_ids(output_input_names);
    auto sync_output = sync_node->get_output();
    output_node->set_input(sync_output, "default");
    graph_->add_node(input_node);
    graph_->add_node(output_node);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to wire I/O nodes: " << e.what() << "\n";
    return false;
  }
}

}  // namespace coalsack
