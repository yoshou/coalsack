#pragma once

#include <fstream>
#include <string>
#include <unordered_set>

#include "graph_proc.h"

namespace coalsack {

// Graphviz DOT format generator for subgraph visualization
class graphviz_exporter {
 public:
  // Export subgraph to DOT format
  static bool export_to_dot(const subgraph& graph, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
      return false;
    }

    file << "digraph coalsack_graph {" << std::endl;
    file << "  rankdir=TB;" << std::endl;
    file << "  node [shape=box, style=rounded];" << std::endl;
    file << std::endl;

    // Track all edges to avoid duplicates
    std::unordered_set<std::string> rendered_edges;

    // Render nodes
    for (uint32_t i = 0; i < graph.get_node_count(); ++i) {
      auto node = graph.get_node(i);
      uint32_t node_id = graph.get_node_id(node.get());

      std::string proc_name = node->get_proc_name();
      std::string label = proc_name + "\\n[" + std::to_string(node_id) + "]";

      // Color based on node type
      std::string color = "lightblue";
      if (proc_name.find("onnx_") == 0) {
        color = "lightgreen";
      } else if (proc_name == "onnx_model_input") {
        color = "yellow";
      } else if (proc_name == "onnx_model_output") {
        color = "orange";
      }

      file << "  node" << node_id << " [label=\"" << label << "\", fillcolor=" << color
           << ", style=filled];" << std::endl;

      // Render output edges
      const auto& outputs = node->get_outputs();
      for (const auto& [output_name, edge] : outputs) {
        std::string edge_label = (output_name == "default") ? "" : output_name;

        // Find nodes that use this edge as input
        for (uint32_t j = 0; j < graph.get_node_count(); ++j) {
          auto other_node = graph.get_node(j);
          const auto& inputs = other_node->get_inputs();

          for (const auto& [input_name, input_edge] : inputs) {
            if (input_edge.get() == edge.get()) {
              uint32_t target_id = graph.get_node_id(other_node.get());

              std::string edge_id = std::to_string(node_id) + "_" + std::to_string(target_id);
              if (rendered_edges.find(edge_id) == rendered_edges.end()) {
                std::string full_label = edge_label;
                if (input_name != "default") {
                  if (!full_label.empty()) full_label += " → ";
                  full_label += input_name;
                }

                file << "  node" << node_id << " -> node" << target_id;
                if (!full_label.empty()) {
                  file << " [label=\"" << full_label << "\"]";
                }
                file << ";" << std::endl;

                rendered_edges.insert(edge_id);
              }
            }
          }
        }
      }
    }

    // Render external inputs (edges with no source)
    for (uint32_t i = 0; i < graph.get_node_count(); ++i) {
      auto node = graph.get_node(i);
      uint32_t node_id = graph.get_node_id(node.get());
      const auto& inputs = node->get_inputs();

      for (const auto& [input_name, edge] : inputs) {
        if (edge->get_source() == nullptr) {
          std::string external_id = "external_" + input_name + "_" + std::to_string(node_id);
          file << "  " << external_id << " [label=\"" << input_name
               << "\", shape=ellipse, fillcolor=pink, style=filled];" << std::endl;
          file << "  " << external_id << " -> node" << node_id << " [label=\"" << input_name
               << "\"];" << std::endl;
        }
      }
    }

    file << "}" << std::endl;
    file.close();

    return true;
  }

  // Export and also generate PNG using dot command (if available)
  static bool export_to_png(const subgraph& graph, const std::string& dot_file,
                            const std::string& png_file) {
    if (!export_to_dot(graph, dot_file)) {
      return false;
    }

    // Try to run dot command
    std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file + " 2>/dev/null";
    int ret = system(cmd.c_str());

    return (ret == 0);
  }
};

}  // namespace coalsack
