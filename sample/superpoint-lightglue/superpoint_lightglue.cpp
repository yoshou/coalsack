#include <fmt/core.h>
#include <onnxruntime_cxx_api.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <boost/asio.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "coalsack/core/graph_proc_client.h"
#include "coalsack/core/graph_proc_server.h"
#include "coalsack/core/subgraph.h"
#include "coalsack/image/frame_message.h"
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/image/image_nodes.h"
#include "coalsack/messages/object_message.h"
#include "coalsack/tensor/graph_proc_tensor.h"

namespace fs = std::filesystem;
using namespace coalsack;
namespace asio = boost::asio;

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown() {
  for (auto handler : on_shutdown_handlers) {
    handler();
  }
  exit_flag.store(true);
}

static void sigint_handler(int) {
  shutdown();
  exit(0);
}

class onnx_runtime_node : public graph_node {
 private:
  std::vector<uint8_t> model_data;
  Ort::Session session;
  std::vector<std::string> input_node_names;
  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};

 public:
  onnx_runtime_node() : graph_node(), session(nullptr) {}

  void set_model_data(const std::vector<uint8_t>& value) { model_data = value; }

  virtual std::string get_proc_name() const override { return "onnx_runtime_node"; }

  template <typename Archive>
  void save(Archive& archive) const {
    std::vector<std::string> output_names;
    auto outputs = get_outputs();
    for (auto output : outputs) {
      output_names.push_back(output.first);
    }
    archive(output_names);
    archive(model_data);
  }

  template <typename Archive>
  void load(Archive& archive) {
    std::vector<std::string> output_names;
    archive(output_names);
    for (auto output_name : output_names) {
      set_output(std::make_shared<graph_edge>(this), output_name);
    }
    archive(model_data);
  }

  graph_edge_ptr add_output(const std::string& name) {
    auto outputs = get_outputs();
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      auto output = std::make_shared<graph_edge>(this);
      set_output(output, name);
      return output;
    }
    return it->second;
  }

  virtual void initialize() override {
    const auto& api = Ort::GetApi();

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // CUDA Provider
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // Create session from memory
    session = Ort::Session(env, model_data.data(), model_data.size(), session_options);

    // Get input node information
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto num_input_nodes = session.GetInputCount();

    for (std::size_t i = 0; i < num_input_nodes; i++) {
      const auto input_name = session.GetInputNameAllocated(i, allocator);
      const auto type_info = session.GetInputTypeInfo(i);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto input_shape = tensor_info.GetShape();

      input_node_names.push_back(input_name.get());
      input_node_dims[input_name.get()] = input_shape;
    }

    // Register output edges for all ONNX outputs
    const auto num_output_nodes = session.GetOutputCount();
    for (std::size_t i = 0; i < num_output_nodes; i++) {
      const auto output_name = session.GetOutputNameAllocated(i, allocator);
      add_output(output_name.get());
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (get_outputs().size() == 0) {
      return;
    }

    // Input: grayscale image pair [2, 1, height, width]
    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(message)) {
      const auto& src = frame_msg->get_data();

      const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      // Prepare input tensors
      std::vector<const char*> input_node_names_cstr;
      std::vector<Ort::Value> input_tensors;

      for (const auto& name : this->input_node_names) {
        input_node_names_cstr.push_back(name.c_str());

        auto dims = input_node_dims.at(name);
        // Update dynamic dimensions
        dims[0] = src.shape[3];  // batch
        dims[2] = src.shape[1];  // height
        dims[3] = src.shape[0];  // width

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(src.data.data()),
                                            src.data.size(), dims.data(), dims.size()));
      }

      // Prepare output node names
      std::vector<const char*> output_node_names_cstr;
      for (const auto& [name, _] : get_outputs()) {
        output_node_names_cstr.push_back(name.c_str());
      }

      // Run ONNX inference
      const auto start = std::chrono::high_resolution_clock::now();
      const auto output_tensors = session.Run(
          Ort::RunOptions{nullptr}, input_node_names_cstr.data(), input_tensors.data(),
          input_tensors.size(), output_node_names_cstr.data(), output_node_names_cstr.size());
      const auto end = std::chrono::high_resolution_clock::now();
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      fmt::print("ONNX inference time: {} ms\n", elapsed);

      // Process output tensors
      assert(output_tensors.size() == output_node_names_cstr.size());

      for (std::size_t i = 0; i < output_node_names_cstr.size(); i++) {
        const auto name = output_node_names_cstr.at(i);
        const auto& value = output_tensors.at(i);

        graph_message_ptr output_msg;

        if (value.IsTensor()) {
          const auto tensor_info = value.GetTensorTypeAndShapeInfo();
          const auto shape = tensor_info.GetShape();
          const auto elem_type = tensor_info.GetElementType();

          fmt::print("Output '{}': shape={}", name, shape.size());
          for (size_t j = 0; j < shape.size(); j++) {
            fmt::print("{}{}", (j == 0 ? "[" : ", "), shape[j]);
          }
          fmt::print("]\n");

          // keypoints [2, 1024, 2] - int64
          if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && shape.size() == 3) {
            const auto data = value.GetTensorData<int64_t>();
            auto msg = std::make_shared<frame_message<tensor<int64_t, 3>>>();
            tensor<int64_t, 3> output_tensor(
                {static_cast<std::uint32_t>(shape.at(2)), static_cast<std::uint32_t>(shape.at(1)),
                 static_cast<std::uint32_t>(shape.at(0))},
                data);
            msg->set_data(std::move(output_tensor));
            msg->set_profile(frame_msg->get_profile());
            msg->set_timestamp(frame_msg->get_timestamp());
            msg->set_frame_number(frame_msg->get_frame_number());
            msg->set_metadata(*frame_msg);
            output_msg = msg;
          }
          // matches [num_matches, 3] - int64
          else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && shape.size() == 2) {
            const auto data = value.GetTensorData<int64_t>();
            auto msg = std::make_shared<frame_message<tensor<int64_t, 2>>>();
            tensor<int64_t, 2> output_tensor(
                {static_cast<std::uint32_t>(shape.at(1)), static_cast<std::uint32_t>(shape.at(0))},
                data);
            msg->set_data(std::move(output_tensor));
            msg->set_profile(frame_msg->get_profile());
            msg->set_timestamp(frame_msg->get_timestamp());
            msg->set_frame_number(frame_msg->get_frame_number());
            msg->set_metadata(*frame_msg);
            output_msg = msg;
          }
          // mscores [num_matches] - float32
          else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && shape.size() == 1) {
            const auto data = value.GetTensorData<float>();
            auto msg = std::make_shared<frame_message<tensor<float, 1>>>();
            tensor<float, 1> output_tensor({static_cast<std::uint32_t>(shape.at(0))}, data);
            msg->set_data(std::move(output_tensor));
            msg->set_profile(frame_msg->get_profile());
            msg->set_timestamp(frame_msg->get_timestamp());
            msg->set_frame_number(frame_msg->get_frame_number());
            msg->set_metadata(*frame_msg);
            output_msg = msg;
          }
        }

        try {
          const auto output = get_output(name);
          output->send(output_msg);
        } catch (const std::exception& e) {
          fmt::print("Error sending output: {}\n", e.what());
        }
      }
    }
  }
};

COALSACK_REGISTER_NODE(onnx_runtime_node, graph_node)

class image_pair_loader_node : public graph_node {
 private:
  std::vector<std::tuple<std::string, std::string>> image_pair_list;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  graph_edge_ptr output;
  graph_edge_ptr output_image0;
  graph_edge_ptr output_image1;

  image_pair_loader_node()
      : graph_node(),
        th(),
        running(false),
        output(std::make_shared<graph_edge>(this)),
        output_image0(std::make_shared<graph_edge>(this)),
        output_image1(std::make_shared<graph_edge>(this)) {
    set_output(output);
    set_output(output_image0, "image0");
    set_output(output_image1, "image1");
  }

  void set_image_pair_list(const std::vector<std::tuple<std::string, std::string>>& value) {
    image_pair_list = value;
  }

  virtual std::string get_proc_name() const override { return "image_pair_loader_node"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(image_pair_list);
  }

  virtual void initialize() override {
    if (image_pair_list.empty()) {
      throw std::runtime_error("Image pair list is empty");
    }
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([&]() {
      uint64_t frame_number = 0;

      for (const auto& [image0_path, image1_path] : image_pair_list) {
        if (!running.load()) break;

        fmt::print("Loading image pair {}: {} and {}\\n", frame_number, image0_path, image1_path);
        cv::Mat img0_bgr = cv::imread(image0_path);
        cv::Mat img1_bgr = cv::imread(image1_path);

        if (img0_bgr.empty() || img1_bgr.empty()) {
          fmt::print("Failed to load image pair {}\\n", frame_number);
          continue;
        }

        // Convert to grayscale
        cv::Mat img0_gray, img1_gray;
        cv::cvtColor(img0_bgr, img0_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img1_bgr, img1_gray, cv::COLOR_BGR2GRAY);

        // Resize if needed
        if (img0_gray.size() != img1_gray.size()) {
          const int common_width = std::min(img0_gray.cols, img1_gray.cols);
          const int common_height = std::min(img0_gray.rows, img1_gray.rows);
          cv::resize(img0_gray, img0_gray, cv::Size(common_width, common_height));
          cv::resize(img1_gray, img1_gray, cv::Size(common_width, common_height));
          cv::resize(img0_bgr, img0_bgr, cv::Size(common_width, common_height));
          cv::resize(img1_bgr, img1_bgr, cv::Size(common_width, common_height));
        }

        const auto width = img0_gray.cols;
        const auto height = img0_gray.rows;

        // Prepare float tensors [W, H, C]
        tensor<float, 3> img0_tensor(
            {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1});
        tensor<float, 3> img1_tensor(
            {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1});

        auto img0_data = img0_tensor.data.data();
        auto img1_data = img1_tensor.data.data();

        // Normalize to [0, 1]
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            img0_data[y * width + x] = img0_gray.at<uint8_t>(y, x) / 255.0f;
            img1_data[y * width + x] = img1_gray.at<uint8_t>(y, x) / 255.0f;
          }
        }

        // Create batched tensor [W, H, C, B]
        tensor<float, 4> batched(
            {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1, 2});

        const auto img_size = width * height;
        auto batched_data = batched.data.data();
        std::memcpy(batched_data, img0_data, img_size * sizeof(float));
        std::memcpy(batched_data + img_size, img1_data, img_size * sizeof(float));

        // Send batched images
        auto output_msg = std::make_shared<frame_message<tensor<float, 4>>>();
        output_msg->set_data(std::move(batched));
        output_msg->set_timestamp(frame_number);
        output_msg->set_frame_number(frame_number);

        output->send(output_msg);

        // Also prepare visualization images
        tensor<uint8_t, 3> img0_vis(
            {static_cast<uint32_t>(img0_bgr.cols), static_cast<uint32_t>(img0_bgr.rows), 3});
        tensor<uint8_t, 3> img1_vis(
            {static_cast<uint32_t>(img1_bgr.cols), static_cast<uint32_t>(img1_bgr.rows), 3});

        std::memcpy(img0_vis.data.data(), img0_bgr.data, img0_bgr.total() * img0_bgr.elemSize());
        std::memcpy(img1_vis.data.data(), img1_bgr.data, img1_bgr.total() * img1_bgr.elemSize());

        auto img0_vis_msg = std::make_shared<frame_message<tensor<uint8_t, 3>>>();
        img0_vis_msg->set_data(std::move(img0_vis));
        img0_vis_msg->set_timestamp(frame_number);
        img0_vis_msg->set_frame_number(frame_number);

        auto img1_vis_msg = std::make_shared<frame_message<tensor<uint8_t, 3>>>();
        img1_vis_msg->set_data(std::move(img1_vis));
        img1_vis_msg->set_timestamp(frame_number);
        img1_vis_msg->set_frame_number(frame_number);

        output_image0->send(img0_vis_msg);
        output_image1->send(img1_vis_msg);

        frame_number++;
      }

      running = false;
    }));
  }

  virtual void stop() override {
    if (running.load()) {
      running.store(false);
      if (th && th->joinable()) {
        th->join();
      }
    }
  }
};

COALSACK_REGISTER_NODE(image_pair_loader_node, graph_node)

class match_visualizer_node : public graph_node {
 private:
  cv::Mat image0, image1;
  tensor<int64_t, 3> keypoints;
  tensor<int64_t, 2> matches;
  tensor<float, 1> mscores;
  int received_count = 0;
  uint64_t frame_number = 0;
  std::string output_dir;

 public:
  match_visualizer_node() : graph_node() {}

  void set_output_dir(const std::string& dir) { output_dir = dir; }

  virtual std::string get_proc_name() const override { return "match_visualizer_node"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(output_dir);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "image0") {
      if (auto frame_msg =
              std::dynamic_pointer_cast<frame_message<tensor<std::uint8_t, 3>>>(message)) {
        const auto& img = frame_msg->get_data();
        const auto w = img.shape[0];
        const auto h = img.shape[1];
        const auto c = img.shape[2];
        image0 = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, const_cast<uint8_t*>(img.data.data()))
                     .clone();
        frame_number = frame_msg->get_frame_number();
        received_count++;
      }
    } else if (input_name == "image1") {
      if (auto frame_msg =
              std::dynamic_pointer_cast<frame_message<tensor<std::uint8_t, 3>>>(message)) {
        const auto& img = frame_msg->get_data();
        const auto w = img.shape[0];
        const auto h = img.shape[1];
        const auto c = img.shape[2];
        image1 = cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, const_cast<uint8_t*>(img.data.data()))
                     .clone();
        received_count++;
      }
    } else if (input_name == "keypoints") {
      if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<int64_t, 3>>>(message)) {
        keypoints = frame_msg->get_data();
        received_count++;
      }
    } else if (input_name == "matches") {
      if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<int64_t, 2>>>(message)) {
        matches = frame_msg->get_data();
        received_count++;
      }
    } else if (input_name == "mscores") {
      if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 1>>>(message)) {
        mscores = frame_msg->get_data();
        received_count++;
      }
    }

    // All inputs received? Visualize!
    if (received_count >= 5) {
      fmt::print("Visualizer received all 5 inputs: keypoints={}, matches={}, mscores={}\n",
                 keypoints.shape[0] * keypoints.shape[1] * keypoints.shape[2],
                 matches.shape[0] * matches.shape[1], mscores.shape[0]);
      visualize();
      received_count = 0;
    }
  }

  void visualize() {
    if (image0.empty() || image1.empty() || matches.shape[1] == 0) {
      fmt::print("Missing data for visualization\n");
      return;
    }

    // Convert grayscale to RGB if needed
    cv::Mat img0_rgb, img1_rgb;
    if (image0.channels() == 1) {
      cv::cvtColor(image0, img0_rgb, cv::COLOR_GRAY2BGR);
    } else {
      img0_rgb = image0;
    }
    if (image1.channels() == 1) {
      cv::cvtColor(image1, img1_rgb, cv::COLOR_GRAY2BGR);
    } else {
      img1_rgb = image1;
    }

    // Create side-by-side image
    const int combined_width = img0_rgb.cols + img1_rgb.cols;
    const int combined_height = std::max(img0_rgb.rows, img1_rgb.rows);
    cv::Mat combined = cv::Mat::zeros(combined_height, combined_width, CV_8UC3);

    img0_rgb.copyTo(combined(cv::Rect(0, 0, img0_rgb.cols, img0_rgb.rows)));
    img1_rgb.copyTo(combined(cv::Rect(img0_rgb.cols, 0, img1_rgb.cols, img1_rgb.rows)));

    // Draw matches
    const auto kpts_data = keypoints.data.data();
    const auto matches_data = matches.data.data();
    const auto scores_data = mscores.data.data();
    const auto num_matches =
        matches.shape[1];  // shape is [3, num_matches] -> num_matches is shape[1]

    fmt::print("Drawing {} matches\n", num_matches);

    for (std::size_t i = 0; i < num_matches; i++) {
      const auto img_idx = matches_data[i * 3 + 0];
      const auto kpt_idx0 = matches_data[i * 3 + 1];
      const auto kpt_idx1 = matches_data[i * 3 + 2];
      const auto score = scores_data[i];

      // Get keypoint coordinates: keypoints shape [2, 1024, 2]
      const auto x0 = kpts_data[0 * 1024 * 2 + kpt_idx0 * 2 + 0];
      const auto y0 = kpts_data[0 * 1024 * 2 + kpt_idx0 * 2 + 1];
      const auto x1 = kpts_data[1 * 1024 * 2 + kpt_idx1 * 2 + 0];
      const auto y1 = kpts_data[1 * 1024 * 2 + kpt_idx1 * 2 + 1];

      cv::Point2f pt0(x0, y0);
      cv::Point2f pt1(x1 + img0_rgb.cols, y1);

      // Color by confidence: green (high) to red (low)
      const int b = static_cast<int>(255 * (1.0f - score));
      const int g = static_cast<int>(255 * score);
      cv::Scalar color(b, g, 0);

      cv::line(combined, pt0, pt1, color, 1, cv::LINE_AA);
      cv::circle(combined, pt0, 3, color, -1, cv::LINE_AA);
      cv::circle(combined, pt1, 3, color, -1, cv::LINE_AA);
    }

    // Calculate statistics
    float mean_score = 0.0f;
    for (std::size_t i = 0; i < num_matches; i++) {
      mean_score += scores_data[i];
    }
    mean_score /= num_matches;

    // Add text overlay
    const std::string text = fmt::format("{} matches, avg score: {:.3f}", num_matches, mean_score);
    cv::putText(combined, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);

    // Save output with frame number
    fs::create_directories(output_dir);
    const auto output_path =
        (fs::path(output_dir) / fmt::format("frame_{:04d}.jpg", frame_number)).string();
    cv::imwrite(output_path, combined);
    fmt::print("Frame {} visualization saved to: {}\n", frame_number, output_path);
  }
};

COALSACK_REGISTER_NODE(match_visualizer_node, graph_node)

class local_server {
  asio::io_context io_context;
  std::shared_ptr<graph_proc_server> server;
  std::unique_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  local_server()
      : io_context(),
        server(std::make_shared<graph_proc_server>(io_context, "0.0.0.0", 31400)),
        th(),
        running(false) {}

  void run() {
    running = true;
    th.reset(new std::thread([this] { io_context.run(); }));
  }

  void stop() {
    if (running.load()) {
      running.store(false);
      io_context.stop();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }

  ~local_server() { stop(); }
};

int main(int argc, char* argv[]) try {
  signal(SIGINT, sigint_handler);

  spdlog::set_level(spdlog::level::debug);

  // Start local server
  local_server server;
  server.run();

  asio::io_context io_context;

  // Load ONNX model
  std::vector<uint8_t> model_data;
  {
    const auto model_path =
        "../sample/superpoint-lightglue/data/superpoint_lightglue_pipeline.onnx";
    std::ifstream ifs;
    ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
    if (ifs.fail()) {
      std::cerr << "File open error: " << model_path << "\n";
      return 1;
    }

    std::istreambuf_iterator<char> ifs_begin(ifs);
    std::istreambuf_iterator<char> ifs_end{};
    std::vector<uint8_t> data(ifs_begin, ifs_end);
    if (ifs.fail()) {
      std::cerr << "File read error: " << model_path << "\n";
      return 1;
    }

    model_data = std::move(data);
    fmt::print("Model loaded: {:.1f} MB\n", model_data.size() / 1024.0 / 1024.0);
  }

  // Create subgraph
  std::shared_ptr<subgraph> g(new subgraph());

  // Image loader node
  std::shared_ptr<image_pair_loader_node> loader(new image_pair_loader_node());
  std::vector<std::tuple<std::string, std::string>> image_pairs = {
      {"../sample/superpoint-lightglue/data/DSC_0410.JPG",
       "../sample/superpoint-lightglue/data/DSC_0411.JPG"},
      {"../sample/superpoint-lightglue/data/sacre_coeur1.jpg",
       "../sample/superpoint-lightglue/data/sacre_coeur2.jpg"}};
  loader->set_image_pair_list(image_pairs);
  g->add_node(loader);

  // ONNX matcher node
  std::shared_ptr<onnx_runtime_node> matcher(new onnx_runtime_node());
  matcher->set_model_data(model_data);
  matcher->set_input(loader->get_output());
  g->add_node(matcher);

  const auto keypoints = matcher->add_output("keypoints");
  const auto matches = matcher->add_output("matches");
  const auto mscores = matcher->add_output("mscores");

  // Visualizer node
  std::shared_ptr<match_visualizer_node> visualizer(new match_visualizer_node());
  visualizer->set_output_dir("output");
  visualizer->set_input(keypoints, "keypoints");
  visualizer->set_input(matches, "matches");
  visualizer->set_input(mscores, "mscores");
  visualizer->set_input(loader->output_image0, "image0");
  visualizer->set_input(loader->output_image1, "image1");
  g->add_node(visualizer);

  // Deploy graph
  graph_proc_client client;
  client.deploy(io_context, "127.0.0.1", 31400, g);

  on_shutdown_handlers.push_back([&client, &server] {
    client.stop();
    server.stop();
  });

  std::thread io_thread([&io_context] { io_context.run(); });

  client.run();

  while (!exit_flag.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  if (io_thread.joinable()) {
    io_thread.join();
  }

  return 0;
} catch (std::exception& e) {
  std::cout << e.what() << std::endl;
  shutdown();
  return 1;
}
