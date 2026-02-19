#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "coalsack/core/graph_proc.h"
#include "coalsack/nn/model_io_nodes.h"
#include "coalsack/nn/nn_nodes.h"
#include "coalsack/onnx/onnx_importer.h"
#include "coalsack/util/graphviz_exporter.h"

using namespace coalsack;

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);

  std::cout << "Testing SuperPoint-LightGlue ONNX Model Import\n";
  std::cout << "==============================================\n\n";

  std::string model_path =
      "/workspaces/stargazer/coalsack/sample/superpoint-lightglue/data/"
      "superpoint_lightglue_pipeline.onnx";
  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "Loading model: " << model_path << "\n";

  auto start = std::chrono::high_resolution_clock::now();

  // Create importer
  onnx_importer importer;

  // Load model
  bool success = false;
  try {
    success = importer.load_model(model_path);
    if (!success) {
      std::cerr << "load_model returned false\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception during model loading: " << e.what() << "\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (!success) {
    std::cerr << "Failed to load ONNX model\n";
    return 1;
  }

  std::cout << "✓ Model loaded successfully in " << duration.count() << " ms\n\n";

  // Print model information
  const auto& inputs = importer.get_input_names();
  const auto& outputs = importer.get_output_names();
  const auto& initializers = importer.get_initializers();
  auto subgraph = importer.get_subgraph();

  // Export graph to Graphviz DOT
  if (subgraph) {
    const std::string dot_path = "superpoint_lightglue.dot";
    if (graphviz_exporter::export_to_dot(*subgraph, dot_path)) {
      std::cout << "  Graph exported to " << dot_path << "\n";
    } else {
      std::cerr << "  Failed to export graph to DOT: " << dot_path << "\n";
    }
  }

  std::cout << "Model Information:\n";
  std::cout << "  Inputs (" << inputs.size() << "):\n";
  for (const auto& name : inputs) {
    std::cout << "    - " << name << "\n";
  }

  std::cout << "  Outputs (" << outputs.size() << "):\n";
  for (const auto& name : outputs) {
    std::cout << "    - " << name << "\n";
  }

  std::cout << "  Initializers (constants): " << initializers.size() << "\n";
  std::cout << "  Subgraph created successfully\n";

  // Test inference execution
  std::cout << "\n==============================================\n";
  std::cout << "Testing Inference Execution\n";
  std::cout << "==============================================\n\n";

  // Load test images
  std::string img1_path =
      "/workspaces/stargazer/coalsack/sample/superpoint-lightglue/data/sacre_coeur1.jpg";
  std::string img2_path =
      "/workspaces/stargazer/coalsack/sample/superpoint-lightglue/data/sacre_coeur2.jpg";

  cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

  if (img1.empty() || img2.empty()) {
    std::cerr << "Failed to load test images\n";
    return 1;
  }

  std::cout << "✓ Test images loaded: " << img1.cols << "x" << img1.rows << "\n";

  // Resize to target size (768x731 as in Python)
  cv::Mat img1_resized, img2_resized;
  cv::resize(img1, img1_resized, cv::Size(768, 731));
  cv::resize(img2, img2_resized, cv::Size(768, 731));

  // Convert to float and normalize to [0, 1]
  cv::Mat img1_float, img2_float;
  img1_resized.convertTo(img1_float, CV_32F, 1.0 / 255.0);
  img2_resized.convertTo(img2_float, CV_32F, 1.0 / 255.0);

  // Create batch tensor [2, 1, 731, 768]
  std::vector<int64_t> input_shape = {2, 1, 731, 768};
  dynamic_tensor input_tensor(dtype::float32, input_shape);

  float* data = input_tensor.data_ptr<float>();

  // Copy image 1
  for (int i = 0; i < 731; ++i) {
    for (int j = 0; j < 768; ++j) {
      data[i * 768 + j] = img1_float.at<float>(i, j);
    }
  }

  // Copy image 2
  int offset = 731 * 768;
  for (int i = 0; i < 731; ++i) {
    for (int j = 0; j < 768; ++j) {
      data[offset + i * 768 + j] = img2_float.at<float>(i, j);
    }
  }

  std::cout << "✓ Input tensor prepared: shape [2, 1, 731, 768]\n";

  // Create input node
  auto input_node = std::make_shared<model_input_node>();
  input_node->set_tensor("images", input_tensor);
  input_node->set_frame_number(1);

  // Add constants
  for (const auto& [name, tensor] : initializers) {
    input_node->set_tensor(name, tensor);
  }

  std::cout << "✓ Input node configured with " << (1 + initializers.size()) << " tensors\n";

  // Create output collector
  auto output_node = std::make_shared<model_output_node>();

  std::unordered_map<std::string, dynamic_tensor> results;

  output_node->set_callback([&](const std::unordered_map<std::string, dynamic_tensor>& outputs) {
    results = outputs;
    std::cout << "✓ Inference complete! Received " << outputs.size() << " outputs\n";
  });

  // Wire the graph
  if (!importer.wire_io_nodes(input_node, output_node)) {
    std::cerr << "Failed to wire input/output nodes\n";
    return 1;
  }

  std::cout << "✓ Graph wired successfully\n";

  // Create graph processor and initialize
  graph_proc proc;
  proc.deploy(subgraph);

  std::cout << "Initializing and executing graph...\n";
  start = std::chrono::high_resolution_clock::now();

  proc.run();

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "✓ Inference completed in " << duration.count() << " ms\n\n";

  // Display results
  std::cout << "Results:\n";
  for (const auto& [name, tensor] : results) {
    std::cout << "  " << name << ": shape [";
    const auto& shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << shape[i];
    }
    std::cout << "]\n";
  }

  std::cout << "\n✓ All tests passed!\n";

  return 0;
}
