#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <boost/asio.hpp>

#include <signal.h>
#include <unistd.h>

#include "graph_proc.h"
#include "graph_proc_img.h"
#include "graph_proc_cv.h"
#include "graph_proc_tensor.h"

#include <fmt/core.h>
#include <cereal/types/array.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

using namespace coalsack;

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown()
{
    for (auto handler : on_shutdown_handlers)
    {
        handler();
    }
    exit_flag.store(true);
}

static void sigint_handler(int)
{
    shutdown();
    exit(0);
}

struct camera_data
{
    double fx;
    double fy;
    double cx;
    double cy;
    std::array<double, 3> k;
    std::array<double, 2> p;
    std::array<std::array<double, 3>, 3> rotation;
    std::array<double, 3> translation;
};

class camera_data_message : public graph_message
{
    camera_data data;

public:
    camera_data_message()
    {
    }

    void set_data(const camera_data &value)
    {
        data = value;
    }
    const camera_data &get_data() const
    {
        return data;
    }
    static std::string get_type()
    {
        return "camera_data";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(data.fx, data.fy, data.cx, data.cy, data.k, data.p, data.rotation, data.translation);
    }
};

class panoptic_data_loader_node : public graph_node
{
    std::string data_dir;
    std::vector<std::string> sequence_list;
    std::vector<std::tuple<int32_t, int32_t>> camera_list;

    std::shared_ptr<std::thread> th;
    std::atomic_bool running;
    graph_edge_ptr output;
    std::vector<std::tuple<std::string, camera_data>> data;

public:
    panoptic_data_loader_node()
        : graph_node(), th(), running(false), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    void set_data_dir(std::string value)
    {
        data_dir = value;
    }

    void set_sequence_list(const std::vector<std::string> &value)
    {
        sequence_list = value;
    }

    void set_camera_list(const std::vector<std::tuple<int32_t, int32_t>>& value)
    {
        camera_list = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "panoptic_data_loader_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(data_dir);
        archive(sequence_list);
        archive(camera_list);
    }

    virtual void initialize() override
    {
        for (const auto& sequence : sequence_list)
        {
            const auto annotation_dir = fs::path(data_dir) / sequence / "hdPose3d_stage1_coco19";

            std::vector<std::string> annotation_files;
            for (const auto &entry : fs::directory_iterator(annotation_dir))
            {
                if (entry.path().extension() == ".json")
                {
                    annotation_files.push_back(entry.path().string());
                }
            }
            std::sort(annotation_files.begin(), annotation_files.end());

            std::map<std::tuple<int32_t, int32_t>, camera_data> cameras;
            {
                const auto camera_file = fs::path(data_dir) / sequence / fmt::format("calibration_{:s}.json", sequence);

                std::ifstream f;
                f.open(camera_file, std::ios::in | std::ios::binary);
                std::string str((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());

                nlohmann::json calib = nlohmann::json::parse(str);

                for (const auto& cam : calib["cameras"])
                {
                    const auto panel = cam["panel"].get<int32_t>();
                    const auto node = cam["node"].get<int32_t>();

                    if (std::find(camera_list.begin(), camera_list.end(), std::make_tuple(panel, node)) == camera_list.end())
                    {
                        continue;
                    }

                    const auto k = cam["K"].get<std::vector<std::vector<double>>>();
                    const auto dist_coeffs = cam["distCoef"].get<std::vector<double>>();
                    const auto rotation = cam["R"].get<std::vector<std::vector<double>>>();
                    const auto translation = cam["t"].get<std::vector<std::vector<double>>>();

                    const std::array<std::array<double, 3>, 3> m = {{
                        {{1.0, 0.0, 0.0}},
                        {{0.0, 0.0, -1.0}},
                        {{0.0, 1.0, 0.0}},
                    }};

                    camera_data cam_data = {};
                    cam_data.fx = k[0][0];
                    cam_data.fy = k[1][1];
                    cam_data.cx = k[0][2];
                    cam_data.cy = k[1][2];
                    for (size_t i = 0; i < 3; i++)
                    {
                        for (size_t j = 0; j < 3; j++)
                        {
                            for (size_t k = 0; k < 3; k++)
                            {
                                cam_data.rotation[i][j] += rotation[i][k] * m[k][j];
                            }
                        }
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        for (size_t j = 0; j < 3; j++)
                        {
                            cam_data.translation[i] += -translation[j][0] * cam_data.rotation[j][i] * 10.0;
                        }
                    }
                    cam_data.k[0] = dist_coeffs[0];
                    cam_data.k[1] = dist_coeffs[1];
                    cam_data.k[2] = dist_coeffs[4];
                    cam_data.p[0] = dist_coeffs[2];
                    cam_data.p[1] = dist_coeffs[3];

                    cameras[std::make_pair(panel, node)] = cam_data;
                }
            }

            for (size_t i = 0; i < annotation_files.size(); i++)
            {
                for (const auto& [camera_panel, camera_node]: camera_list)
                {
                    const auto prefix = fmt::format("{:02d}_{:02d}", camera_panel, camera_node);
                    std::string postfix = fs::path(annotation_files[i]).filename().string();
                    const std::string to_erase = "body3DScene";
                    postfix.erase(postfix.find(to_erase), to_erase.size());
                    const auto image_file = (fs::path(data_dir) / sequence / "hdImgs" / prefix / (prefix + postfix)).replace_extension(".jpg").string();
                    const auto camera = cameras.at(std::make_tuple(camera_panel, camera_node));
                    data.push_back(std::make_tuple(image_file, camera));
                }
            }
        }
    }

    virtual void run() override
    {
        running = true;
        th.reset(new std::thread([&]()
        {
            const auto num_view = camera_list.size();
            uint64_t frame_number = 0;

            for (size_t i = 0; running.load() && i < data.size(); i += num_view, ++frame_number)
            {
                auto msg = std::make_shared<object_message>();

                for (size_t j = 0; j < num_view; j++)
                {
                    const auto [image_file, camera] = data[i + j];
                    const auto [camera_panel, camera_node] = camera_list[j];
                    const auto camera_name = fmt::format("camera_{:02d}_{:02d}", camera_panel, camera_node);

                    auto data = cv::imread(image_file, cv::IMREAD_UNCHANGED | cv::IMREAD_IGNORE_ORIENTATION);
                    cv::cvtColor(data, data, cv::COLOR_BGR2RGB);

                    const auto get_scale = [](const cv::Size2f& image_size, const cv::Size2f& resized_size) {
                        float w_pad, h_pad;
                        if (image_size.width / resized_size.width < image_size.height / resized_size.height)
                        {
                            w_pad = image_size.height / resized_size.height * resized_size.width;
                            h_pad = image_size.height;
                        }
                        else
                        {
                            w_pad = image_size.width;
                            h_pad = image_size.width / resized_size.width * resized_size.height;
                        }

                        return cv::Size2f(w_pad / 200.0, h_pad / 200.0);
                    };

                    const auto&& image_size = cv::Size2f(960, 512);
                    const auto scale = get_scale(data.size(), image_size);
                    const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);

                    const auto get_tri_3rd_point = [](const cv::Point2f& a, const cv::Point2f& b) {
                        const auto direct = a - b;
                        return b + cv::Point2f(-direct.y, direct.x);
                    };

                    const auto get_affine_transform = [&](const cv::Point2f& center, const cv::Size2f& scale, const cv::Size2f& output_size) {
                        const auto src_w = scale.width * 200.0;
                        const auto src_h = scale.height * 200.0;
                        const auto dst_w = output_size.width;
                        const auto dst_h = output_size.height;

                        cv::Point2f src_dir, dst_dir;
                        if (src_w >= src_h)
                        {
                            src_dir = cv::Point2f(0, src_w * -0.5);
                            dst_dir = cv::Point2f(0, dst_w * -0.5);
                        }
                        else
                        {
                            src_dir = cv::Point2f(src_h * -0.5, 0);
                            dst_dir = cv::Point2f(dst_h * -0.5, 0);
                        }

                        const auto src_tri_a = center;
                        const auto src_tri_b = center + src_dir;
                        const auto src_tri_c = get_tri_3rd_point(src_tri_a, src_tri_b);
                        cv::Point2f src_tri[3] = {src_tri_a, src_tri_b, src_tri_c};

                        const auto dst_tri_a = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
                        const auto dst_tri_b = dst_tri_a + dst_dir;
                        const auto dst_tri_c = get_tri_3rd_point(dst_tri_a, dst_tri_b);
                        
                        cv::Point2f dst_tri[3] = {dst_tri_a, dst_tri_b, dst_tri_c};

                        cv::Mat trans = cv::getAffineTransform(src_tri, dst_tri);

                        return trans;
                    };

                    const auto trans = get_affine_transform(center, scale, image_size);

                    cv::Mat input_img;
                    cv::warpAffine(data, input_img, trans, cv::Size(image_size), cv::INTER_LINEAR);

                    tensor<uint8_t, 4> input_img_tensor({static_cast<std::uint32_t>(input_img.size().width),
                                                         static_cast<std::uint32_t>(input_img.size().height),
                                                         static_cast<std::uint32_t>(input_img.elemSize()),
                                                         1},
                                                        (const uint8_t *)input_img.data,
                                                        {static_cast<std::uint32_t>(input_img.step[1]),
                                                         static_cast<std::uint32_t>(input_img.step[0]),
                                                         static_cast<std::uint32_t>(1),
                                                         static_cast<std::uint32_t>(input_img.total())});

                    auto camera_data_msg = std::make_shared<camera_data_message>();
                    camera_data_msg->set_data(camera);

                    auto frame_msg = std::make_shared<frame_message<tensor<float, 4>>>();
    
                    const auto input_img_tensor_f = input_img_tensor.cast<float>().transform([this](const float value, const size_t w, const size_t h, const size_t c, const size_t n)
                                                { return value / 255.0f; });

                    frame_msg->set_data(std::move(input_img_tensor_f));
                    frame_msg->set_timestamp(0);
                    frame_msg->set_frame_number(frame_number);
                    frame_msg->set_metadata("camera", camera_data_msg);

                    msg->add_field(camera_name, frame_msg);
                }

                output->send(msg);
            }

            running = false;
        }));
    }

    virtual void stop() override
    {
        if (running.load())
        {
            running.store(false);
            if (th && th->joinable())
            {
                th->join();
            }
        }
    }
};

CEREAL_REGISTER_TYPE(panoptic_data_loader_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, panoptic_data_loader_node)

class object_map_node : public graph_node
{
public:
    object_map_node()
        : graph_node()
    {
    }

    virtual std::string get_proc_name() const override
    {
        return "object_map_node";
    }

    template <typename Archive>
    void save(Archive &archive) const
    {
        std::vector<std::string> output_names;
        auto outputs = get_outputs();
        for (auto output : outputs)
        {
            output_names.push_back(output.first);
        }
        archive(output_names);
    }

    template <typename Archive>
    void load(Archive &archive)
    {
        std::vector<std::string> output_names;
        archive(output_names);
        for (auto output_name : output_names)
        {
            set_output(std::make_shared<graph_edge>(this), output_name);
        }
    }

    graph_edge_ptr add_output(const std::string& name)
    {
        auto outputs = get_outputs();
        auto it = outputs.find(name);
        if (it == outputs.end())
        {
            auto output = std::make_shared<graph_edge>(this);
            set_output(output, name);
            return output;
        }
        return it->second;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            for (const auto &[name, field] : obj_msg->get_fields())
            {
                try
                {
                    const auto output = get_output(name);
                    output->send(field);
                }
                catch(const std::exception& e)
                {
                    spdlog::error(e.what());
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(object_map_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, object_map_node)

class normalize_node : public graph_node
{
    std::vector<float> mean;
    std::vector<float> std;
    graph_edge_ptr output;

public:
    normalize_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    void set_mean(const std::vector<float> &value)
    {
        mean = value;
    }

    void set_std(const std::vector<float> &value)
    {
        std = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "normalize_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(mean);
        archive(std);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(message))
        {
            const auto &src = frame_msg->get_data();

            const auto dst = src.transform([this](const float value, const size_t w, const size_t h, const size_t c, const size_t n)
                                           { return (value - mean[c]) / std[c]; });

            auto msg = std::make_shared<frame_message<tensor<float, 4>>>();

            msg->set_data(std::move(dst));
            msg->set_profile(frame_msg->get_profile());
            msg->set_timestamp(frame_msg->get_timestamp());
            msg->set_frame_number(frame_msg->get_frame_number());
            msg->set_metadata(*frame_msg);

            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(normalize_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, normalize_node)

class onnx_runtime_node : public graph_node
{
    std::vector<uint8_t> model_data;

    Ort::Session session;
    std::vector<std::string> input_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;

public:
    onnx_runtime_node()
        : graph_node(), session(nullptr)
    {
    }

    void set_model_data(const std::vector<uint8_t> &value)
    {
        model_data = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "onnx_runtime_node";
    }

    template <typename Archive>
    void save(Archive &archive) const
    {
        std::vector<std::string> output_names;
        auto outputs = get_outputs();
        for (auto output : outputs)
        {
            output_names.push_back(output.first);
        }
        archive(output_names);
        archive(model_data);
    }

    template <typename Archive>
    void load(Archive &archive)
    {
        std::vector<std::string> output_names;
        archive(output_names);
        for (auto output_name : output_names)
        {
            set_output(std::make_shared<graph_edge>(this), output_name);
        }
        archive(model_data);
    }

    graph_edge_ptr add_output(const std::string &name)
    {
        auto outputs = get_outputs();
        auto it = outputs.find(name);
        if (it == outputs.end())
        {
            auto output = std::make_shared<graph_edge>(this);
            set_output(output, name);
            return output;
        }
        return it->second;
    }

    Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "test"};

    virtual void initialize() override
    {
        const auto &api = Ort::GetApi();

        // Create session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.arena_extend_strategy = 0;
        options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        options.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(options);

        session = Ort::Session(env, model_data.data(), model_data.size(), session_options);

        // Iterate over all input nodes
        const size_t num_input_nodes = session.GetInputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            const auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());

            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            const auto type = tensor_info.GetElementType();
            
            const auto input_shape = tensor_info.GetShape();
            input_node_dims[input_name.get()] = input_shape;
        }
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (get_outputs().size() == 0)
        {
            return;
        }

        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(message))
        {
            const auto &src = frame_msg->get_data();

            const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<const char *> input_node_names;
            std::vector<Ort::Value> input_tensors;
            for (const auto &name : this->input_node_names)
            {
                input_node_names.push_back(name.c_str());

                const auto dims = input_node_dims.at(name);
                input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(src.get_data()), src.get_size(),
                                                                        dims.data(), dims.size()));
            }

            std::vector<const char *> output_node_names;
            for (const auto &[name, _] : get_outputs())
            {
                output_node_names.push_back(name.c_str());
            }

            const auto output_tensors =
                session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), output_node_names.size());

            assert(output_tensors.size() == output_node_names.size());
            for (std::size_t i = 0; i < output_node_names.size(); i++)
            {
                const auto name = output_node_names.at(i);
                const auto& value = output_tensors.at(i);

                graph_message_ptr output_msg;

                if (value.IsTensor())
                {
                    auto msg = std::make_shared<frame_message<tensor<float, 4>>>();

                    const auto data = value.GetTensorData<float>();
                    const auto tensor_info = value.GetTensorTypeAndShapeInfo();
                    const auto type = tensor_info.GetElementType();
                    const auto shape = tensor_info.GetShape();

                    tensor<float, 4> output_tensor({static_cast<std::uint32_t>(shape.at(3)),
                                                    static_cast<std::uint32_t>(shape.at(2)),
                                                    static_cast<std::uint32_t>(shape.at(1)),
                                                    static_cast<std::uint32_t>(shape.at(0))},
                                                   data);

                    msg->set_data(std::move(output_tensor));
                    msg->set_profile(frame_msg->get_profile());
                    msg->set_timestamp(frame_msg->get_timestamp());
                    msg->set_frame_number(frame_msg->get_frame_number());
                    msg->set_metadata(*frame_msg);

                    output_msg = msg;
                }

                try
                {
                    const auto output = get_output(name);
                    output->send(output_msg);
                }
                catch (const std::exception &e)
                {
                    spdlog::error(e.what());
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(onnx_runtime_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, onnx_runtime_node)

class project_node : public graph_node
{
    graph_edge_ptr output;

public:
    project_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "project_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            for (const auto &[name, field] : obj_msg->get_fields())
            {
                if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(field))
                {
                    const auto &src = frame_msg->get_data();
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(project_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, project_node)

using tensor_f32_4_frame_number_sync_node = sync_node<tensor_f32_4, frame_number_sync_config>;

CEREAL_REGISTER_TYPE(tensor_f32_4_frame_number_sync_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, tensor_f32_4_frame_number_sync_node)

class local_server
{
    asio::io_service io_service;
    std::shared_ptr<graph_proc_server> server;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;

public:
    local_server()
        : io_service(), server(std::make_shared<graph_proc_server>(io_service, "0.0.0.0", 31400)), th(), running(false)
    {
    }

    void run()
    {
        running = true;
        th.reset(new std::thread([this]
                                 { io_service.run(); }));
    }

    void stop()
    {
        if (running.load())
        {
            running.store(false);
            io_service.stop();
            if (th && th->joinable())
            {
                th->join();
            }
        }
    }

    ~local_server()
    {
        stop();
    }
};

int main(int argc, char *argv[])
try
{
    signal(SIGINT, sigint_handler);

    spdlog::set_level(spdlog::level::debug);

    local_server server;
    server.run();

    asio::io_service io_service;

    std::shared_ptr<subgraph> g(new subgraph());

    std::vector<std::tuple<int32_t, int32_t>> camera_list = {
        {0, 12},
        {0, 6},
        {0, 23},
        {0, 13},
        {0, 3},
    };

    std::vector<uint8_t> backbone_model_data;
    {
        const auto model_path = "../sample/dnn/data/backbone.onnx";
        std::ifstream ifs;
        ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
        if (ifs.fail())
        {
            std::cerr << "File open error: " << model_path << "\n";
            std::quick_exit(0);
        }

        std::istreambuf_iterator<char> ifs_begin(ifs);
        std::istreambuf_iterator<char> ifs_end{};
        std::vector<uint8_t> data(ifs_begin, ifs_end);
        if (ifs.fail())
        {
            std::cerr << "File read error: " << model_path << "\n";
            std::quick_exit(0);
        }

        backbone_model_data = std::move(data);
    }

    std::shared_ptr<panoptic_data_loader_node> data_loader(new panoptic_data_loader_node());
    data_loader->set_data_dir("/workspace/panoptic-toolbox/data");
    data_loader->set_sequence_list({"171204_pose1"});
    data_loader->set_camera_list(camera_list);
    g->add_node(data_loader);

    std::shared_ptr<object_map_node> map_data(new object_map_node());
    map_data->set_input(data_loader->get_output());
    g->add_node(map_data);

    std::unordered_map<std::string, graph_edge_ptr> heatmaps_list;
    for (const auto& [camera_panel, camera_node] : camera_list)
    {
        const auto camera_name = fmt::format("camera_{:02d}_{:02d}", camera_panel, camera_node);
        const auto camera_image = map_data->add_output(camera_name);

        std::shared_ptr<normalize_node> normalize(new normalize_node());
        normalize->set_input(camera_image);
        normalize->set_mean({0.485, 0.456, 0.406});
        normalize->set_std({0.229, 0.224, 0.225});
        g->add_node(normalize);

        const auto normalized = normalize->get_output();

        std::shared_ptr<onnx_runtime_node> inference_backbone(new onnx_runtime_node());
        inference_backbone->set_input(normalized);
        inference_backbone->set_model_data(backbone_model_data);
        g->add_node(inference_backbone);

        const auto heatmaps = inference_backbone->add_output("output");
        heatmaps_list[camera_name] = heatmaps;
    }

    std::shared_ptr<tensor_f32_4_frame_number_sync_node> sync(new tensor_f32_4_frame_number_sync_node());
    for (const auto &[camera_name, heatmaps] : heatmaps_list)
    {
        sync->set_input(heatmaps, camera_name);
    }
    g->add_node(sync);

    std::shared_ptr<project_node> project(new project_node());
    project->set_input(sync->get_output());
    g->add_node(project);

    graph_proc_client client;
    client.deploy(io_service, "127.0.0.1", 31400, g);

    on_shutdown_handlers.push_back([&client, &server]
                                   {
        client.stop();
        server.stop(); });

    std::thread io_thread([&io_service]
                          { io_service.run(); });

    client.run();

    while (!exit_flag.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    if (io_thread.joinable())
    {
        io_thread.join();
    }

    return 0;
}
catch (std::exception &e)
{
    std::cout << e.what() << std::endl;
    shutdown();
}