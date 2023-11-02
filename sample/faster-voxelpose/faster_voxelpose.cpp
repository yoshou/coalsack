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
#include <opencv2/calib3d/calib3d.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef ENABLE_TFLITE_EXT
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#endif

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

struct roi_data
{
    std::array<double, 2> scale;
    double rotation;
    std::array<double, 2> center;
};

class roi_data_message : public graph_message
{
    roi_data data;

public:
    roi_data_message()
    {
    }

    void set_data(const roi_data &value)
    {
        data = value;
    }
    const roi_data &get_data() const
    {
        return data;
    }
    static std::string get_type()
    {
        return "roi_data";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(data.scale, data.rotation, data.center);
    }
};

static cv::Mat get_transform(const cv::Point2f &center, const cv::Size2f &scale, const cv::Size2f &output_size)
{
    const auto get_tri_3rd_point = [](const cv::Point2f &a, const cv::Point2f &b)
    {
        const auto direct = a - b;
        return b + cv::Point2f(-direct.y, direct.x);
    };

    const auto get_affine_transform = [&](const cv::Point2f &center, const cv::Size2f &scale, const cv::Size2f &output_size)
    {
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

        return cv::getAffineTransform(src_tri, dst_tri);
    };

    return get_affine_transform(center, scale, output_size);
}

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
                    const auto rotation = 0.0;

                    const auto trans = get_transform(center, scale, image_size);

                    const auto input_img = data;

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

                    auto roi_data_msg = std::make_shared<roi_data_message>();
                    roi_data_msg->set_data({{scale.width, scale.height}, rotation, {center.x, center.y}});

                    auto frame_msg = std::make_shared<frame_message<tensor<float, 4>>>();
    
                    const auto input_img_tensor_f = input_img_tensor.cast<float>().transform([this](const float value, const size_t w, const size_t h, const size_t c, const size_t n)
                                                { return value / 255.0f; });

                    frame_msg->set_data(std::move(input_img_tensor_f));
                    frame_msg->set_timestamp(0);
                    frame_msg->set_frame_number(frame_number);
                    frame_msg->set_metadata("camera", camera_data_msg);
                    frame_msg->set_metadata("roi", roi_data_msg);

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
            for (const auto& [name, output] : get_outputs())
            {
                const auto &field = obj_msg->get_field(name);
                try
                {
                    output->send(field);
                }
                catch (const std::exception &e)
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

class onnx_runtime_session
{
public:
    Ort::Session session;
    std::vector<std::string> input_names;
    std::unordered_map<std::string, std::vector<int64_t>> input_dims;
    std::unordered_map<std::string, int> input_types;

    static Ort::Session create_session(const Ort::Env &env, const std::vector<uint8_t> &model_data)
    {
        const auto &api = Ort::GetApi();

        // Create session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.arena_extend_strategy = 0;
        options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        options.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(options);

        return Ort::Session(env, model_data.data(), model_data.size(), session_options);
    }

    onnx_runtime_session(const Ort::Env &env, const std::vector<uint8_t> &model_data)
        : session(create_session(env, model_data))
    {
        // Iterate over all input nodes
        const size_t num_input_nodes = session.GetInputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            const auto input_name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.get());

            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            const auto type = tensor_info.GetElementType();
            input_types[input_name.get()] = type;

            const auto input_shape = tensor_info.GetShape();
            input_dims[input_name.get()] = input_shape;
        }
    }
};

class onnx_runtime_session_pool
{
    struct vector_hash
    {
        size_t operator()(const std::vector<uint8_t> &v) const
        {
            size_t hash = v.size();
            for (const auto i : v)
                hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            return hash;
        }
    };

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
    std::unordered_map<std::vector<uint8_t>, std::shared_ptr<onnx_runtime_session>, vector_hash> sessions;
    std::mutex mtx;

public:
    std::shared_ptr<onnx_runtime_session> get_or_load(const std::vector<uint8_t>& model_data)
    {
        std::lock_guard<std::mutex> lock(mtx);

        auto it = sessions.find(model_data);
        if (it == sessions.end())
        {
            auto session = std::make_shared<onnx_runtime_session>(env, model_data);
            sessions[model_data] = session;
            return session;
        }
        return it->second;
    }
};

class onnx_runtime_node : public graph_node
{
    std::vector<uint8_t> model_data;
    std::shared_ptr<onnx_runtime_session> session;

    static onnx_runtime_session_pool sessions;

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

    virtual void initialize() override
    {
        session = sessions.get_or_load(model_data);
    }

    template <size_t num_dims, typename T, std::size_t... Is>
    static std::shared_ptr<frame_message_base> create_message(const std::vector<int64_t> &shape, const T *data, const frame_message_base *base_msg, std::index_sequence<Is...>)
    {
        auto msg = std::make_shared<frame_message<tensor<T, num_dims>>>();
        tensor<T, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(num_dims - 1 - Is))...}, data);

        msg->set_data(std::move(output_tensor));
        msg->set_profile(base_msg->get_profile());
        msg->set_timestamp(base_msg->get_timestamp());
        msg->set_frame_number(base_msg->get_frame_number());
        msg->set_metadata(*base_msg);

        return msg;
    }

    template <size_t num_dims, typename T>
    static std::shared_ptr<frame_message_base> create_message(const std::vector<int64_t> &shape, const T *data, const frame_message_base *base_msg)
    {
        if (shape.size() == num_dims)
        {
            return create_message<num_dims>(shape, data, base_msg, std::make_index_sequence<num_dims>{});
        }
        if constexpr (num_dims > 1)
        {
            return create_message<num_dims - 1>(shape, data, base_msg);
        }
        else
        {
            return nullptr;
        }
    }

    template <size_t num_dims>
    static std::tuple<size_t, const float *, std::shared_ptr<frame_message_base>> get_input_data(const graph_message_ptr &msg)
    {
        if (const auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, num_dims>>>(msg))
        {
            const auto &src = frame_msg->get_data();
            const auto size = src.get_size();
            const auto data = src.get_data();

            return std::make_tuple(size, data, std::dynamic_pointer_cast<frame_message_base>(msg));
        }
    
        if constexpr (num_dims > 1)
        {
            return get_input_data<num_dims - 1>(msg);
        }
        else
        {
            return std::make_tuple(static_cast<size_t>(0), nullptr, std::shared_ptr<frame_message_base>());
        }
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (get_outputs().size() == 0)
        {
            return;
        }

        static std::mutex mtx;

        std::unordered_map<std::string, graph_message_ptr> output_msgs;

        {
            std::lock_guard<std::mutex> lock(mtx);

            std::vector<const char *> output_name_strs;
            std::vector<Ort::Value> output_tensors;
            std::shared_ptr<frame_message_base> base_frame_msg = nullptr;

            {
                const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                std::vector<const char *> input_name_strs;
                std::vector<Ort::Value> input_tensors;

                if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(message))
                {
                    assert(session->input_names.size() == 1);
                    {
                        const auto name = session->input_names.at(0);
                        const auto dims = session->input_dims.at(name);
                        const auto type = session->input_types.at(name);
                        assert(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

                        input_name_strs.push_back(name.c_str());

                        size_t expect_size = 1;
                        for (const auto dim : dims)
                        {
                            expect_size *= dim;
                        }

                        const float *data = nullptr;
                        size_t size = 0;
                        std::tie(size, data, base_frame_msg) = get_input_data<5>(frame_msg);

                        assert(size == expect_size);
                        assert(data != nullptr);
                        assert(base_frame_msg != nullptr);

                        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(data), size,
                                                                                dims.data(), dims.size()));
                    }
                }

                if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
                {
                    for (const auto &name : session->input_names)
                    {
                        const auto dims = session->input_dims.at(name);
                        const auto type = session->input_types.at(name);
                        assert(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

                        input_name_strs.push_back(name.c_str());

                        size_t expect_size = 1;
                        for (const auto dim : dims)
                        {
                            expect_size *= dim;
                        }

                        const float* data = nullptr;
                        size_t size = 0;
                        std::tie(size, data, base_frame_msg) = get_input_data<5>(obj_msg->get_field(name));

                        assert(size == expect_size);
                        assert(data != nullptr);
                        assert(base_frame_msg != nullptr);

                        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(data), size,
                                                                                dims.data(), dims.size()));
                    }
                }

                assert(input_name_strs.size() == session->input_names.size());
                assert(input_tensors.size() == session->input_names.size());

                for (const auto &[name, _] : get_outputs())
                {
                    output_name_strs.push_back(name.c_str());
                }

                output_tensors = session->session.Run(Ort::RunOptions{nullptr}, input_name_strs.data(), input_tensors.data(), input_tensors.size(), output_name_strs.data(), output_name_strs.size());
            }

            assert(output_tensors.size() == output_name_strs.size());
            for (std::size_t i = 0; i < output_name_strs.size(); i++)
            {
                const auto name = output_name_strs.at(i);
                const auto &value = output_tensors.at(i);

                graph_message_ptr output_msg;

                if (value.IsTensor())
                {
                    const auto data = value.GetTensorData<float>();
                    const auto tensor_info = value.GetTensorTypeAndShapeInfo();
                    const auto type = tensor_info.GetElementType();
                    const auto shape = tensor_info.GetShape();

                    assert(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

                    output_msg = create_message<5>(shape, data, base_frame_msg.get());
                }

                assert(output_msg);

                output_msgs[name] = output_msg;
            }
        }

        for (const auto &[name, output_msg] : output_msgs)
        {
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
};

onnx_runtime_session_pool onnx_runtime_node::sessions;

CEREAL_REGISTER_TYPE(onnx_runtime_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, onnx_runtime_node)

#ifdef ENABLE_TFLITE_EXT
#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

class tensorflow_lite_node : public graph_node
{
    std::vector<uint8_t> model_data;

    Ort::Session session;
    std::vector<std::string> input_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;

public:
    tensorflow_lite_node()
        : graph_node(), session(nullptr)
    {
    }

    void set_model_data(const std::vector<uint8_t> &value)
    {
        model_data = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "tensorflow_lite_node";
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

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    virtual void initialize() override
    {
        model = tflite::FlatBufferModel::BuildFromBuffer(reinterpret_cast<const char*>(model_data.data()), model_data.size());
        TFLITE_MINIMAL_CHECK(model != nullptr);

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        TFLITE_MINIMAL_CHECK(interpreter != nullptr);

        TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
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

            for (size_t input = 0; input < interpreter->inputs().size(); input++)
            {
                const auto input_tensor = interpreter->input_tensor(input);
                auto input_data = interpreter->typed_input_tensor<float>(input);

                assert(input_tensor->dims->size == 4);

                tensor<float, 4>::view_type input_tensor_view;
                input_tensor_view.data = input_data;
                for (size_t dim = 0; dim < 4; dim++)
                {
                    input_tensor_view.shape[dim] = input_tensor->dims->data[input_tensor->dims->size - 1 - dim];
                }
                input_tensor_view.stride[0] = 1;
                for (size_t dim = 1; dim < 4; dim++)
                {
                    input_tensor_view.stride[dim] = input_tensor_view.stride[dim - 1] * input_tensor_view.shape[dim - 1];
                }

                input_tensor_view.assign(src.transpose({2, 0, 1, 3}), [](auto, const auto& value, auto...) { return value; });
            }

            TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

            for (size_t output = 0; output < interpreter->outputs().size(); output++)
            {
                const auto output_tensor = interpreter->output_tensor(output);
                const auto output_data = interpreter->typed_input_tensor<float>(output);
                const auto name = output_tensor->name;

                graph_message_ptr output_msg;

                if (output_tensor->dims->size == 4)
                {
                    constexpr auto num_dims = 4;

                    std::array<uint32_t, num_dims> output_tensor_shape;
                    for (size_t dim = 0; dim < num_dims; dim++)
                    {
                        output_tensor_shape[dim] = output_tensor->dims->data[output_tensor->dims->size - 1 - dim];
                    }
                    std::array<uint32_t, num_dims> output_tensor_stride;
                    output_tensor_stride[0] = 1;
                    for (size_t dim = 1; dim < num_dims; dim++)
                    {
                        output_tensor_stride[dim] = output_tensor_stride[dim - 1] * output_tensor_shape[dim - 1];
                    }

                    auto msg = std::make_shared<frame_message<tensor<float, num_dims>>>();
                    tensor<float, num_dims> tensor({static_cast<std::uint32_t>(output_tensor_shape[1]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[2]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[0]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[3])},
                                                   output_data,
                                                   {static_cast<std::uint32_t>(output_tensor_stride[1]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[2]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[0]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[3])});

                    msg->set_data(std::move(tensor));
                    msg->set_profile(frame_msg->get_profile());
                    msg->set_timestamp(frame_msg->get_timestamp());
                    msg->set_frame_number(frame_msg->get_frame_number());
                    msg->set_metadata(*frame_msg);

                    output_msg = msg;
                }
                else if (output_tensor->dims->size == 5)
                {
                    constexpr auto num_dims = 5;

                    std::array<uint32_t, num_dims> output_tensor_shape;
                    for (size_t dim = 0; dim < num_dims; dim++)
                    {
                        output_tensor_shape[dim] = output_tensor->dims->data[output_tensor->dims->size - 1 - dim];
                    }
                    std::array<uint32_t, num_dims> output_tensor_stride;
                    output_tensor_stride[0] = 1;
                    for (size_t dim = 1; dim < num_dims; dim++)
                    {
                        output_tensor_stride[dim] = output_tensor_stride[dim - 1] * output_tensor->dims->data[output_tensor->dims->size - 1 - (dim - 1)];
                    }

                    auto msg = std::make_shared<frame_message<tensor<float, num_dims>>>();
                    tensor<float, num_dims> tensor({static_cast<std::uint32_t>(output_tensor_shape[1]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[2]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[0]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[3]),
                                                    static_cast<std::uint32_t>(output_tensor_shape[4])},
                                                   output_data,
                                                   {static_cast<std::uint32_t>(output_tensor_stride[1]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[2]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[0]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[3]),
                                                    static_cast<std::uint32_t>(output_tensor_stride[4])});

                    msg->set_data(std::move(tensor));
                    msg->set_profile(frame_msg->get_profile());
                    msg->set_timestamp(frame_msg->get_timestamp());
                    msg->set_frame_number(frame_msg->get_frame_number());
                    msg->set_metadata(*frame_msg);

                    output_msg = msg;
                }

                assert(output_msg);

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

CEREAL_REGISTER_TYPE(tensorflow_lite_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, tensorflow_lite_node)
#endif

class pre_project_whole_node : public graph_node
{
    graph_edge_ptr output;

    std::array<float, 3> grid_center;

public:
    pre_project_whole_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "pre_project_whole_node";
    }

    std::array<float, 3> get_grid_center() const
    {
        return grid_center;
    }
    void set_grid_center(const std::array<float, 3> &value)
    {
        grid_center = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(grid_center);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            tensor<float, 1> grid_center({3});
            grid_center.set({0}, this->grid_center[0]);
            grid_center.set({1}, this->grid_center[1]);
            grid_center.set({2}, this->grid_center[2]);

            std::map<std::string, graph_message_ptr> ordered_fields;
            for (const auto &[name, field] : obj_msg->get_fields())
            {
                ordered_fields.insert(std::make_pair(name, field));
            }

            using heatmaps_list_type = std::vector<std::tuple<tensor<float, 4>, camera_data, roi_data>>;

            heatmaps_list_type heatmaps;

            std::shared_ptr<frame_message<tensor<float, 4>>> src_msg = nullptr;
            for (const auto &[name, field] : ordered_fields)
            {
                if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(field))
                {
                    auto &src = frame_msg->get_data();

                    const auto camera_msg = frame_msg->get_metadata<camera_data_message>("camera");
                    const auto camera = camera_msg->get_data();

                    const auto roi_msg = frame_msg->get_metadata<roi_data_message>("roi");
                    const auto roi = roi_msg->get_data();
                    
                    heatmaps.emplace_back(src, camera, roi);

                    src_msg = frame_msg;
                }
            }

            if (src_msg == nullptr)
            {
                return;
            }

            auto msg = std::make_shared<object_message>();

            auto grid_center_msg = std::make_shared<frame_message<tensor<float, 1>>>();
            grid_center_msg->set_data(grid_center);
            grid_center_msg->set_profile(src_msg->get_profile());
            grid_center_msg->set_timestamp(src_msg->get_timestamp());
            grid_center_msg->set_frame_number(src_msg->get_frame_number());
            grid_center_msg->set_metadata(*src_msg);

            auto heatmaps_list_msg = std::make_shared<frame_message<heatmaps_list_type>>();
            heatmaps_list_msg->set_data(heatmaps);
            heatmaps_list_msg->set_profile(src_msg->get_profile());
            heatmaps_list_msg->set_timestamp(src_msg->get_timestamp());
            heatmaps_list_msg->set_frame_number(src_msg->get_frame_number());
            heatmaps_list_msg->set_metadata(*src_msg);

            msg->add_field("grid_center", grid_center_msg);
            msg->add_field("heatmaps", heatmaps_list_msg);

            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(pre_project_whole_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, pre_project_whole_node)

class project_whole_node : public graph_node
{
    graph_edge_ptr output;

    std::array<float, 3> grid_size;
    std::array<int32_t, 3> cube_size;

public:
    project_whole_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "project_whole_node";
    }

    std::array<float, 3> get_grid_size() const
    {
        return grid_size;
    }
    void set_grid_size(const std::array<float, 3>& value)
    {
        grid_size = value;
    }
    std::array<int32_t, 3> get_cube_size() const
    {
        return cube_size;
    }
    void set_cube_size(const std::array<int32_t, 3>& value)
    {
        cube_size = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(grid_size, cube_size);
    }

    std::vector<std::array<float, 3>> compute_grid(const std::array<float, 3>& grid_center) const
    {
        std::vector<std::array<float, 3>> grid;
        for (int32_t x = 0; x < cube_size.at(0); x++)
        {
            for (int32_t y = 0; y < cube_size.at(1); y++)
            {
                for (int32_t z = 0; z < cube_size.at(2); z++)
                {
                    const auto gridx = -grid_size.at(0) / 2 + grid_size.at(0) * x / (cube_size.at(0) - 1) + grid_center.at(0);
                    const auto gridy = -grid_size.at(1) / 2 + grid_size.at(1) * y / (cube_size.at(1) - 1) + grid_center.at(1);
                    const auto gridz = -grid_size.at(2) / 2 + grid_size.at(2) * z / (cube_size.at(2) - 1) + grid_center.at(2);

                    grid.push_back({gridx, gridy, gridz});
                }
            }
        }
        return grid;
    }

    static std::vector<std::array<float, 2>> project_point(const std::vector<std::array<float, 3>>& x, const camera_data& camera)
    {
        std::vector<cv::Point3d> points;

        std::transform(x.begin(), x.end(), std::back_inserter(points), [&](const auto& p) {
            const auto pt_x = p[0] - camera.translation[0];
            const auto pt_y = p[1] - camera.translation[1];
            const auto pt_z = p[2] - camera.translation[2];
            const auto cam_x = pt_x * camera.rotation[0][0] + pt_y * camera.rotation[0][1] + pt_z * camera.rotation[0][2];
            const auto cam_y = pt_x * camera.rotation[1][0] + pt_y * camera.rotation[1][1] + pt_z * camera.rotation[1][2];
            const auto cam_z = pt_x * camera.rotation[2][0] + pt_y * camera.rotation[2][1] + pt_z * camera.rotation[2][2];

            return cv::Point3d(cam_x / (cam_z + 1e-5), cam_y / (cam_z + 1e-5), 1.0);
        });

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);
        camera_matrix.at<double>(0, 0) = camera.fx;
        camera_matrix.at<double>(1, 1) = camera.fy;
        camera_matrix.at<double>(0, 2) = camera.cx;
        camera_matrix.at<double>(1, 2) = camera.cy;

        cv::Mat dist_coeffs(5, 1, cv::DataType<double>::type);
        dist_coeffs.at<double>(0) = camera.k[0];
        dist_coeffs.at<double>(1) = camera.k[1];
        dist_coeffs.at<double>(2) = camera.p[0];
        dist_coeffs.at<double>(3) = camera.p[1];
        dist_coeffs.at<double>(4) = camera.k[2];

        cv::Mat rvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
        cv::Mat tvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);

        std::vector<cv::Point2d> projected_points;
        cv::projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);

        std::vector<std::array<float, 2>> y;

        std::transform(projected_points.begin(), projected_points.end(), std::back_inserter(y), [](const auto& p) {
            return std::array<float, 2>{static_cast<float>(p.x), static_cast<float>(p.y)};
        });

        return y;
    }

    static tensor<float, 4> grid_sample(const tensor<float, 4> &src, const std::vector<std::array<float, 2>>& grid, bool align_corner = false)
    {
        const auto num_o = src.shape[3];
        const auto num_c = src.shape[2];
        const auto num_h = src.shape[1];
        const auto num_w = src.shape[0];

        tensor<float, 4> dst({static_cast<uint32_t>(grid.size()), 1, num_c, num_o});

        constexpr size_t num_size = SHRT_MAX - 1;

        for (size_t offset = 0; offset < grid.size(); offset += num_size)
        {
            const auto grid_num = std::min(num_size, grid.size() - offset);

            cv::Mat map_x(grid_num, 1, cv::DataType<float>::type);
            cv::Mat map_y(grid_num, 1, cv::DataType<float>::type);

            if (align_corner)
            {
                for (size_t i = 0; i < grid_num; i++)
                {
                    const auto x = ((grid[i + offset][0] + 1) / 2) * (num_w - 1);
                    const auto y = ((grid[i + offset][1] + 1) / 2) * (num_h - 1);
                    map_x.at<float>(i, 0) = x;
                    map_y.at<float>(i, 0) = y;
                }
            }
            else
            {
                for (size_t i = 0; i < grid_num; i++)
                {
                    const auto x = ((grid[i + offset][0] + 1) * num_w - 1) / 2;
                    const auto y = ((grid[i + offset][1] + 1) * num_h - 1) / 2;
                    map_x.at<float>(i, 0) = x;
                    map_y.at<float>(i, 0) = y;
                }
            }

            for (uint32_t o = 0; o < num_o; o++)
            {
                for (uint32_t c = 0; c < num_c; c++)
                {
                    cv::Mat plane(num_h, num_w, cv::DataType<float>::type, const_cast<float *>(src.get_data()) + c * src.stride[2] + o * src.stride[3]);
                    cv::Mat remapped(grid_num, 1, cv::DataType<float>::type, dst.get_data() + offset + c * dst.stride[2] + o * dst.stride[3]);
                    cv::remap(plane, remapped, map_x, map_y, cv::INTER_LINEAR);
                }
            }
        }

        return dst;
    }

    tensor<float, 4> get_voxel(const std::vector<tensor<float, 4>> &heatmaps, const std::vector<camera_data> &cameras, const std::vector<roi_data> &rois, const std::array<float, 3> &grid_center) const
    {
        const auto num_bins = std::accumulate(cube_size.begin(), cube_size.end(), 1, std::multiplies<int32_t>());
        const auto num_joints = heatmaps.at(0).shape[2];
        const auto num_cameras = heatmaps.size();
        const auto w = heatmaps.at(0).shape[0];
        const auto h = heatmaps.at(0).shape[1];
        const auto grid = compute_grid(grid_center);

        auto cubes = tensor<float, 4>::zeros({num_cameras, num_bins, 1, num_joints});

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto &roi = rois.at(c);
            const auto &&image_size = cv::Size2f(960, 512);
            const auto center = cv::Point2f(roi.center[0], roi.center[1]);
            const auto scale = cv::Size2f(roi.scale[0], roi.scale[1]);
            const auto width = center.x * 2;
            const auto height = center.y * 2;

            const auto trans = get_transform(center, scale, image_size);
            cv::Mat transf;
            trans.convertTo(transf, cv::DataType<float>::type);

            const auto xy = project_point(grid, cameras[c]);

            std::vector<std::array<float, 2>> sample_grid;
            std::transform(xy.begin(), xy.end(), std::back_inserter(sample_grid), [&](const auto& p) {
                const auto x0 = p[0];
                const auto y0 = p[1];

                const auto x1 = std::clamp(x0, -1.0f, std::max(width, height));
                const auto y1 = std::clamp(y0, -1.0f, std::max(width, height));

                const auto x2 = x1 * transf.at<float>(0, 0) + y1 * transf.at<float>(0, 1) + transf.at<float>(0, 2);
                const auto y2 = x1 * transf.at<float>(1, 0) + y1 * transf.at<float>(1, 1) + transf.at<float>(1, 2);

                const auto x3 = x2 * w / image_size.width;
                const auto y3 = y2 * h / image_size.height;

                const auto x4 = x3 / (w - 1) * 2.0f - 1.0f;
                const auto y4 = y3 / (h - 1) * 2.0f - 1.0f;

                const auto x5 = std::clamp(x4, -1.1f, 1.1f);
                const auto y5 = std::clamp(y4, -1.1f, 1.1f);

                return std::array<float, 2>{x5, y5};
            });

            const auto cube = grid_sample(heatmaps[c], sample_grid, true);

            auto camera_cubes = cubes.view({0, cubes.shape[1], cubes.shape[2], cubes.shape[3]}, {c, 0, 0, 0});

            camera_cubes.assign(cube.view(), [](const float value1, const float value2, auto...)
                                { return value1 + value2; });
        }

        const auto merged_cubes = cubes
            .sum<1>({0}).view()
            .transform(
                [num_cameras](const float value, auto...) {
                    return std::clamp(value / num_cameras, 0.f, 1.f);
                });

        const auto output_cubes = merged_cubes.view<4>({cube_size[2], cube_size[1], cube_size[0], num_joints}).contiguous();
        return output_cubes;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            std::shared_ptr<frame_message_base> base_frame_msg = nullptr;

            using heatmaps_list_type = std::vector<std::tuple<tensor<float, 4>, camera_data, roi_data>>;

            heatmaps_list_type heatmaps_and_metas;
            tensor<float, 1> grid_center;

            for (const auto &[name, field] : obj_msg->get_fields())
            {
                if (name == "heatmaps")
                {
                    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<heatmaps_list_type>>(field))
                    {
                        heatmaps_and_metas = frame_msg->get_data();
                        base_frame_msg = frame_msg;
                    }
                }
                else if (name == "grid_center")
                {
                    if (const auto grid_center_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 1>>>(field))
                    {
                        grid_center = grid_center_msg->get_data();
                    }
                }
            }

            std::vector<tensor<float, 4>> heatmaps;
            std::vector<camera_data> cameras;
            std::vector<roi_data> rois;

            for (const auto& [heatmap, camera, roi] : heatmaps_and_metas)
            {
                heatmaps.push_back(heatmap);
                cameras.push_back(camera);
                rois.push_back(roi);
            }

            if (heatmaps.size() == 0)
            {
                return;
            }

            const auto cubes = get_voxel(heatmaps, cameras, rois, {grid_center.get({0}), grid_center.get({1}), grid_center.get({2})});

            auto msg = std::make_shared<frame_message<tensor<float, 4>>>();

            msg->set_data(std::move(cubes));
            msg->set_profile(base_frame_msg->get_profile());
            msg->set_timestamp(base_frame_msg->get_timestamp());
            msg->set_frame_number(base_frame_msg->get_frame_number());
            msg->set_metadata(*base_frame_msg);

            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(project_whole_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, project_whole_node)

class center_decode_node : public graph_node
{
    graph_edge_ptr output;
    uint32_t max_num;

public:
    center_decode_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "center_decode_node";
    }

    void set_max_num(uint32_t value)
    {
        max_num = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(max_num);
    }

    static tensor<float, 4> max_pool(const tensor<float, 4>& inputs, size_t kernel = 3)
    {
        const auto padding = (kernel - 1) / 2;
        const auto max = inputs.max_pool3d(kernel, 1, padding, 1);
        const auto keep = inputs
            .transform(max,
                [](const float value1, const float value2, auto...) {
                    return value1 == value2 ? value1 : 0.f;
                });
        return keep;
    }

    static tensor<uint64_t, 2> get_index_2d(const tensor<uint64_t, 1> &indices, const std::array<uint64_t, 2>& shape)
    {
        const auto num_people = indices.shape[3];
        const auto result = indices
            .transform_expand<1>({2},
                [shape](const uint64_t value, auto...) {
                    const auto index_x = value / shape[0];
                    const auto index_y = value % shape[0];
                    return std::array<uint64_t, 2>{index_x, index_y};
                });
        return result;
    }

    static std::tuple<tensor<float, 1>, tensor<uint64_t, 2>, tensor<uint64_t, 1>> nms2d(const tensor<float, 4> &hm, const size_t max_num)
    {
        const auto nms = max_pool(hm);
        const auto [topk_values, topk_index] = nms.view<1>({hm.shape[0] * hm.shape[1] * hm.shape[2]})
                                                   .topk(max_num);
        const auto topk_unravel_index = get_index_2d(topk_index, {hm.shape[0], hm.shape[1]});

        return std::forward_as_tuple(topk_values, topk_unravel_index, topk_index);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto base_frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("hm"));

            const auto &hm = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(obj_msg->get_field("hm"))->get_data();
            const auto &size = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(obj_msg->get_field("size"))->get_data();
            const auto &feats = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(obj_msg->get_field("feats"))->get_data();

            const auto [topk_2d_confs, topk_2d_index, topk_2d_flatten_index] = nms2d(hm, max_num);

            const auto flatten_size = size.view<2>({size.shape[0] * size.shape[1], size.shape[2]});

            tensor<float, 2> topk_size({2, topk_2d_flatten_index.shape[0]});
            for (uint32_t i = 0; i < topk_2d_flatten_index.shape[0]; i++)
            {
                topk_size.set({0, i}, flatten_size.get({topk_2d_flatten_index.get({i}), 0}));
                topk_size.set({1, i}, flatten_size.get({topk_2d_flatten_index.get({i}), 1}));
            }

            const auto flatten_feats = feats.view<3>({feats.shape[0], feats.shape[1] * feats.shape[2], feats.shape[3]});

            tensor<float, 3> topk_feats({feats.shape[0], feats.shape[3], topk_2d_flatten_index.shape[0]});
            for (uint32_t i = 0; i < topk_2d_flatten_index.shape[0]; i++)
            {
                for (uint32_t j = 0; j < feats.shape[0]; j++)
                {
                    for (uint32_t k = 0; k < feats.shape[3]; k++)
                    {
                        topk_feats.set({j, k, i}, flatten_feats.get({j, topk_2d_flatten_index.get({i}), k}));
                    }
                }
            }

            auto new_obj_msg = std::make_shared<object_message>();

            const auto make_message = [base_frame_msg](const auto &data)
            {
                auto msg = std::make_shared<frame_message<std::decay_t<decltype(data)>>>();
                msg->set_data(std::move(data));
                msg->set_profile(base_frame_msg->get_profile());
                msg->set_timestamp(base_frame_msg->get_timestamp());
                msg->set_frame_number(base_frame_msg->get_frame_number());
                msg->set_metadata(*base_frame_msg);
                return msg;
            };

            new_obj_msg->add_field("topk_2d_confs", make_message(topk_2d_confs));
            new_obj_msg->add_field("topk_2d_index", make_message(topk_2d_index));
            new_obj_msg->add_field("topk_size", make_message(topk_size));
            new_obj_msg->add_field("topk_feats", make_message(topk_feats));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(center_decode_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, center_decode_node)

class proposal_node : public graph_node
{
    graph_edge_ptr output;
    uint32_t max_num;
    float threshold;
    std::array<float, 3> grid_size;
    std::array<float, 3> grid_center;
    std::array<int32_t, 3> cube_size;

public:
    proposal_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "proposal_node";
    }

    void set_max_num(uint32_t value)
    {
        max_num = value;
    }
    void set_threshold(float value)
    {
        threshold = value;
    }
    std::array<float, 3> get_grid_size() const
    {
        return grid_size;
    }
    void set_grid_size(const std::array<float, 3> &value)
    {
        grid_size = value;
    }
    std::array<float, 3> get_grid_center() const
    {
        return grid_center;
    }
    void set_grid_center(const std::array<float, 3> &value)
    {
        grid_center = value;
    }
    std::array<int32_t, 3> get_cube_size() const
    {
        return cube_size;
    }
    void set_cube_size(const std::array<int32_t, 3> &value)
    {
        cube_size = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(max_num, threshold, grid_size, grid_center, cube_size);
    }

    tensor<float, 2> get_real_loc(const tensor<uint64_t, 2>& index)
    {
        const auto loc = index.cast<float>()
            .transform(
                [this](const float value, const size_t i, const size_t j) {
                    return value / (cube_size[i] - 1) * grid_size[i] + grid_center[i] - grid_size[i] / 2.0f;
                });
        return loc;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("topk_2d_confs"));

            const auto &topk_2d_confs = std::dynamic_pointer_cast<frame_message<tensor<float, 1>>>(obj_msg->get_field("topk_2d_confs"))->get_data();
            const auto &topk_2d_index = std::dynamic_pointer_cast<frame_message<tensor<uint64_t, 2>>>(obj_msg->get_field("topk_2d_index"))->get_data();
            const auto &topk_size = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("topk_size"))->get_data();
            const auto &feat_1d = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(obj_msg->get_field("feat_1d"))->get_data();

            const auto [topk_1d_confs, topk_1d_index] = feat_1d.view<2>({feat_1d.shape[0], feat_1d.shape[1] * feat_1d.shape[2]}).topk(1);

            const auto topk_index = tensor<uint64_t, 2>::concat<0>({topk_2d_index, topk_1d_index});

            const auto topk_confs = topk_2d_confs.view().transform(topk_1d_confs.squeeze(0), [](const auto &value1, const auto &value2, auto...)
                                                            { return value1 * value2; });

            const auto topk_loc = get_real_loc(topk_index);

            auto proposal_centers = tensor<float, 2>::zeros({7, max_num});
            proposal_centers.view({3, proposal_centers.shape[1]}, {0, 0})
                .assign(topk_loc.view(), [](auto, const float value, auto...) {
                    return value;
                });
            proposal_centers.view<1>({0, proposal_centers.shape[1]}, {4, 0})
                .assign(topk_confs.view(), [](auto, const float value, auto...) {
                    return value;
                });
            proposal_centers.view<1>({0, proposal_centers.shape[1]}, {3, 0})
                .assign(topk_confs.view(), [this](auto, const float value, auto...) {
                    return (value > threshold ? 1.f : 0.f) - 1.f;
                });
            proposal_centers.view({2, proposal_centers.shape[1]}, {5, 0})
                .assign(topk_size.view(), [this](auto, const float value, auto...) {
                    return value;
                });

            auto msg = std::make_shared<frame_message<tensor<float, 2>>>();

            msg->set_data(std::move(proposal_centers));
            msg->set_profile(frame_msg->get_profile());
            msg->set_timestamp(frame_msg->get_timestamp());
            msg->set_frame_number(frame_msg->get_frame_number());
            msg->set_metadata(*frame_msg);

            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(proposal_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, proposal_node)

class iterate_proposal_node : public graph_node
{
    graph_edge_ptr output;

public:
    iterate_proposal_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "iterate_proposal_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            using heatmaps_list_type = std::vector<std::tuple<tensor<float, 4>, camera_data, roi_data>>;

            std::shared_ptr<frame_message<heatmaps_list_type>> heatmaps_msg = nullptr;
            std::shared_ptr<frame_message<tensor<float, 2>>> proposal_msg = nullptr;

            for (const auto &[name, field] : obj_msg->get_fields())
            {
                if (name == "heatmaps")
                {
                    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<heatmaps_list_type>>(field))
                    {
                        heatmaps_msg = frame_msg;
                    }
                }
                if (name == "proposal")
                {
                    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(field))
                    {
                        proposal_msg = frame_msg;
                    }
                }
            }

            if (heatmaps_msg == nullptr || proposal_msg == nullptr)
            {
                return;
            }

            auto &proposal = proposal_msg->get_data();

            for (uint32_t i = 0; i < proposal.shape[1]; i++)
            {
                if (proposal.get({3, i}) >= 0)
                {
                    auto msg = std::make_shared<object_message>();

                    auto proposal_center_msg = std::make_shared<frame_message<tensor<float, 1>>>();
                    proposal_center_msg->set_data(std::move(proposal.view<1>({proposal.shape[0], 0}, {0, i}).contiguous()));
                    proposal_center_msg->set_profile(proposal_msg->get_profile());
                    proposal_center_msg->set_timestamp(proposal_msg->get_timestamp());
                    proposal_center_msg->set_frame_number(proposal_msg->get_frame_number());
                    proposal_center_msg->set_metadata(*proposal_msg);

                    msg->add_field("proposal_center", proposal_center_msg);
                    msg->add_field("heatmaps", heatmaps_msg);

                    output->send(msg);
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(iterate_proposal_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, iterate_proposal_node)

class project_individual_node : public graph_node
{
    graph_edge_ptr output;

    std::array<float, 3> grid_size;
    std::array<float, 3> individual_grid_size;
    std::array<int32_t, 3> cube_size;
    std::array<float, 3> grid_center;

    std::array<int32_t, 3> fine_cube_size;

public:
    project_individual_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "project_individual_node";
    }

    std::array<float, 3> get_grid_size() const
    {
        return grid_size;
    }
    void set_grid_size(const std::array<float, 3>& value)
    {
        grid_size = value;
    }
    std::array<float, 3> get_individual_grid_size() const
    {
        return individual_grid_size;
    }
    void set_individual_grid_size(const std::array<float, 3> &value)
    {
        individual_grid_size = value;
    }
    std::array<int32_t, 3> get_cube_size() const
    {
        return cube_size;
    }
    void set_cube_size(const std::array<int32_t, 3>& value)
    {
        cube_size = value;
    }

    std::array<float, 3> get_grid_center() const
    {
        return grid_center;
    }
    void set_grid_center(const std::array<float, 3> &value)
    {
        grid_center = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(grid_size, individual_grid_size, cube_size, grid_center);
    }
    virtual void initialize() override
    {
        for (size_t i = 0; i < 3; i++)
        {
            fine_cube_size[i] = static_cast<int32_t>(grid_size[i] / individual_grid_size[i] * (cube_size[i] - 1)) + 1;
        }
    }

    static std::vector<std::array<float, 3>> compute_grid(const std::array<float, 3> &grid_size, const std::array<float, 3> &grid_center, const std::array<int32_t, 3> &cube_size)
    {
        std::vector<std::array<float, 3>> grid;
        for (int32_t x = 0; x < cube_size.at(0); x++)
        {
            for (int32_t y = 0; y < cube_size.at(1); y++)
            {
                for (int32_t z = 0; z < cube_size.at(2); z++)
                {
                    const auto gridx = -grid_size.at(0) / 2 + grid_size.at(0) * x / (cube_size.at(0) - 1) + grid_center.at(0);
                    const auto gridy = -grid_size.at(1) / 2 + grid_size.at(1) * y / (cube_size.at(1) - 1) + grid_center.at(1);
                    const auto gridz = -grid_size.at(2) / 2 + grid_size.at(2) * z / (cube_size.at(2) - 1) + grid_center.at(2);

                    grid.push_back({gridx, gridy, gridz});
                }
            }
        }
        return grid;
    }

    static std::vector<std::array<float, 2>> project_point(const std::vector<std::array<float, 3>>& x, const camera_data& camera)
    {
        std::vector<cv::Point3d> points;

        std::transform(x.begin(), x.end(), std::back_inserter(points), [&](const auto& p) {
            const auto pt_x = p[0] - camera.translation[0];
            const auto pt_y = p[1] - camera.translation[1];
            const auto pt_z = p[2] - camera.translation[2];
            const auto cam_x = pt_x * camera.rotation[0][0] + pt_y * camera.rotation[0][1] + pt_z * camera.rotation[0][2];
            const auto cam_y = pt_x * camera.rotation[1][0] + pt_y * camera.rotation[1][1] + pt_z * camera.rotation[1][2];
            const auto cam_z = pt_x * camera.rotation[2][0] + pt_y * camera.rotation[2][1] + pt_z * camera.rotation[2][2];

            return cv::Point3d(cam_x / (cam_z + 1e-5), cam_y / (cam_z + 1e-5), 1.0);
        });

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);
        camera_matrix.at<double>(0, 0) = camera.fx;
        camera_matrix.at<double>(1, 1) = camera.fy;
        camera_matrix.at<double>(0, 2) = camera.cx;
        camera_matrix.at<double>(1, 2) = camera.cy;

        cv::Mat dist_coeffs(5, 1, cv::DataType<double>::type);
        dist_coeffs.at<double>(0) = camera.k[0];
        dist_coeffs.at<double>(1) = camera.k[1];
        dist_coeffs.at<double>(2) = camera.p[0];
        dist_coeffs.at<double>(3) = camera.p[1];
        dist_coeffs.at<double>(4) = camera.k[2];

        cv::Mat rvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
        cv::Mat tvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);

        std::vector<cv::Point2d> projected_points;
        cv::projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);

        std::vector<std::array<float, 2>> y;

        std::transform(projected_points.begin(), projected_points.end(), std::back_inserter(y), [](const auto& p) {
            return std::array<float, 2>{static_cast<float>(p.x), static_cast<float>(p.y)};
        });

        return y;
    }

    static tensor<float, 4> grid_sample(const tensor<float, 4> &src, const std::vector<std::array<float, 2>>& grid, bool align_corner = false)
    {
        const auto num_o = src.shape[3];
        const auto num_c = src.shape[2];
        const auto num_h = src.shape[1];
        const auto num_w = src.shape[0];

        tensor<float, 4> dst({static_cast<uint32_t>(grid.size()), 1, num_c, num_o});

        constexpr size_t num_size = SHRT_MAX - 1;

        for (size_t offset = 0; offset < grid.size(); offset += num_size)
        {
            const auto grid_num = std::min(num_size, grid.size() - offset);

            cv::Mat map_x(grid_num, 1, cv::DataType<float>::type);
            cv::Mat map_y(grid_num, 1, cv::DataType<float>::type);

            if (align_corner)
            {
                for (size_t i = 0; i < grid_num; i++)
                {
                    const auto x = ((grid[i + offset][0] + 1) / 2) * (num_w - 1);
                    const auto y = ((grid[i + offset][1] + 1) / 2) * (num_h - 1);
                    map_x.at<float>(i, 0) = x;
                    map_y.at<float>(i, 0) = y;
                }
            }
            else
            {
                for (size_t i = 0; i < grid_num; i++)
                {
                    const auto x = ((grid[i + offset][0] + 1) * num_w - 1) / 2;
                    const auto y = ((grid[i + offset][1] + 1) * num_h - 1) / 2;
                    map_x.at<float>(i, 0) = x;
                    map_y.at<float>(i, 0) = y;
                }
            }

            for (uint32_t o = 0; o < num_o; o++)
            {
                for (uint32_t c = 0; c < num_c; c++)
                {
                    cv::Mat plane(num_h, num_w, cv::DataType<float>::type, const_cast<float *>(src.get_data()) + c * src.stride[2] + o * src.stride[3]);
                    cv::Mat remapped(grid_num, 1, cv::DataType<float>::type, dst.get_data() + offset + c * dst.stride[2] + o * dst.stride[3]);
                    cv::remap(plane, remapped, map_x, map_y, cv::INTER_LINEAR);
                }
            }
        }

        return dst;
    }

    std::tuple<tensor<float, 4>, std::array<float, 3>> get_voxel(const std::vector<tensor<float, 4>> &heatmaps, const std::vector<camera_data> &cameras, const std::vector<roi_data> &rois, const std::array<float, 3> &center, const std::array<float, 2> &size) const
    {
        const auto num_bins = std::accumulate(fine_cube_size.begin(), fine_cube_size.end(), 1, std::multiplies<int32_t>());
        const auto num_joints = heatmaps.at(0).shape[2];
        const auto num_cameras = heatmaps.size();
        const auto w = heatmaps.at(0).shape[0];
        const auto h = heatmaps.at(0).shape[1];
        const auto grid = compute_grid(grid_size, grid_center, fine_cube_size);

        auto cubes = tensor<float, 5>::zeros({num_cameras, cube_size[2], cube_size[1], cube_size[0], num_joints});

        std::array<float, 3> scale, bias;
        for (size_t i = 0; i < 3; i++)
        {
            scale[i] = (fine_cube_size[i] - 1.0f) / grid_size[i];
            bias[i] = -individual_grid_size[i] / 2.0f / grid_size[i] * (fine_cube_size[i] - 1) - scale[i] * (grid_center[i] - grid_size[i] / 2.0f);
        }

        std::array<float, 3> offset;
        std::array<int32_t, 3> center_tl;
        for (size_t i = 0; i < 3; i++)
        {
            center_tl[i] = static_cast<int32_t>(std::round(center[i] * scale[i] + bias[i]));
            offset[i] = static_cast<float>(center_tl[i]) / (fine_cube_size[i] - 1) * grid_size[i] - grid_size[i] / 2.0f + individual_grid_size[i] / 2.0f;
        }

        std::array<int32_t, 3> mask;
        for (size_t i = 0; i < 2; i++)
        {
            mask[i] = std::max(0, static_cast<int32_t>((1 - size[i]) / 2.0f * (cube_size[i] - 1)));
        }
        mask[2] = 0;

        std::array<int32_t, 3> start, end;
        for (size_t i = 0; i < 3; i++)
        {
            start[i] = std::max(0, center_tl[i] + mask[i]);
            end[i] = std::min(fine_cube_size[i], center_tl[i] + cube_size[i] - mask[i]);
        }

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto &roi = rois.at(c);
            const auto &&image_size = cv::Size2f(960, 512);
            const auto center = cv::Point2f(roi.center[0], roi.center[1]);
            const auto scale = cv::Size2f(roi.scale[0], roi.scale[1]);
            const auto width = center.x * 2;
            const auto height = center.y * 2;

            const auto trans = get_transform(center, scale, image_size);
            cv::Mat transf;
            trans.convertTo(transf, cv::DataType<float>::type);

            const auto xy = project_point(grid, cameras[c]);

            std::vector<std::array<float, 2>> sample_grid;
            std::transform(xy.begin(), xy.end(), std::back_inserter(sample_grid), [&](const auto& p) {
                const auto x0 = p[0];
                const auto y0 = p[1];

                const auto x1 = std::clamp(x0, -1.0f, std::max(width, height));
                const auto y1 = std::clamp(y0, -1.0f, std::max(width, height));

                const auto x2 = x1 * transf.at<float>(0, 0) + y1 * transf.at<float>(0, 1) + transf.at<float>(0, 2);
                const auto y2 = x1 * transf.at<float>(1, 0) + y1 * transf.at<float>(1, 1) + transf.at<float>(1, 2);

                const auto x3 = x2 * w / image_size.width;
                const auto y3 = y2 * h / image_size.height;

                const auto x4 = x3 / (w - 1) * 2.0f - 1.0f;
                const auto y4 = y3 / (h - 1) * 2.0f - 1.0f;

                const auto x5 = std::clamp(x4, -1.1f, 1.1f);
                const auto y5 = std::clamp(y4, -1.1f, 1.1f);

                return std::array<float, 2>{x5, y5};
            });

            std::vector<std::array<float, 2>> individual_sample_grid;
            for (size_t x = start[0]; x < end[0]; x++)
            {
                for (size_t y = start[1]; y < end[1]; y++)
                {
                    for (size_t z = start[2]; z < end[2]; z++)
                    {
                        individual_sample_grid.push_back(sample_grid[z + y * fine_cube_size[2] + x * fine_cube_size[2] * fine_cube_size[1]]);
                    }
                }
            }

            const auto cube = grid_sample(heatmaps[c], individual_sample_grid, true).view<4>({end[2] - start[2], end[1] - start[1], end[0] - start[0], num_joints});

            auto camera_cubes = cubes.view<4>({0, end[2] - start[2], end[1] - start[1], end[0] - start[0], num_joints}, {c, start[2] - center_tl[2], start[1] - center_tl[1], start[0] - center_tl[0], 0});

            camera_cubes.assign(cube, [](const float value1, const float value2, auto...)
                                { return value1 + value2; });
        }

        const auto merged_cubes = cubes
            .sum<1>({0}).view()
            .transform(
                [num_cameras](const float value, auto...) {
                    return std::clamp(value / num_cameras, 0.f, 1.f);
                });

        const auto output_cubes = merged_cubes.view<4>({cube_size[2], cube_size[1], cube_size[0], num_joints}).contiguous();
        return std::forward_as_tuple(output_cubes, offset);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            std::shared_ptr<frame_message_base> base_frame_msg = nullptr;

            using heatmaps_list_type = std::vector<std::tuple<tensor<float, 4>, camera_data, roi_data>>;

            heatmaps_list_type heatmaps_and_metas;
            tensor<float, 1> proposal_center;

            for (const auto &[name, field] : obj_msg->get_fields())
            {
                if (name == "heatmaps")
                {
                    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<heatmaps_list_type>>(field))
                    {
                        heatmaps_and_metas = frame_msg->get_data();
                        base_frame_msg = frame_msg;
                    }
                }
                else if (name == "proposal_center")
                {
                    if (const auto proposal_center_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 1>>>(field))
                    {
                        proposal_center = proposal_center_msg->get_data();
                    }
                }
            }

            std::vector<tensor<float, 4>> heatmaps;
            std::vector<camera_data> cameras;
            std::vector<roi_data> rois;

            for (const auto& [heatmap, camera, roi] : heatmaps_and_metas)
            {
                heatmaps.push_back(heatmap);
                cameras.push_back(camera);
                rois.push_back(roi);
            }

            if (heatmaps.size() == 0)
            {
                return;
            }

            const auto [cubes, offset] = get_voxel(heatmaps, cameras, rois, {proposal_center.get({0}), proposal_center.get({1}), proposal_center.get({2})}, {proposal_center.get({5}), proposal_center.get({6})});

            const auto grid = compute_grid(individual_grid_size, grid_center, cube_size);

            std::array<uint32_t, 4> grid_shape = {3, cube_size[2], cube_size[1], cube_size[0]};
            std::array<uint32_t, 4> grid_stride = {1};
            for (size_t i = 1; i < grid_shape.size(); i++)
            {
                grid_stride[i] = grid_stride[i - 1] * grid_shape[i - 1];
            }

            tensor<float, 4>::const_view_type grid_view;
            grid_view.data = grid.data()->data();
            grid_view.shape = grid_shape;
            grid_view.stride = grid_stride;

            const auto center_grid = tensor<float, 2>::stack({
                grid_view.view<3>({2, 0, grid_view.shape[2], grid_view.shape[3]}, {0, 0, 0, 0}).reshape<2>({2, grid_view.shape[2] * grid_view.shape[3]}),
                grid_view.view<3>({2, grid_view.shape[1], 0, grid_view.shape[3]}, {0, 0, 0, 0}, {2, 1, 1, 1}).reshape<2>({2, grid_view.shape[1] * grid_view.shape[3]}),
                grid_view.view<3>({2, grid_view.shape[1], grid_view.shape[2], 0}, {1, 0, 0, 0}).reshape<2>({2, grid_view.shape[1] * grid_view.shape[2]}),
            });

            auto grid_msg = std::make_shared<frame_message<tensor<float, 3>>>();
            grid_msg->set_data(std::move(center_grid));

            auto offset_msg = std::make_shared<frame_message<std::array<float, 3>>>();
            offset_msg->set_data(std::move(offset));

            auto msg = std::make_shared<frame_message<tensor<float, 4>>>();

            msg->set_data(std::move(cubes));
            msg->set_profile(base_frame_msg->get_profile());
            msg->set_timestamp(base_frame_msg->get_timestamp());
            msg->set_frame_number(base_frame_msg->get_frame_number());
            msg->set_metadata(*base_frame_msg);
            msg->set_metadata("grid", grid_msg);
            msg->set_metadata("offset", offset_msg);

            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(project_individual_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, project_individual_node)

class project_orthogonal_node : public graph_node
{
    graph_edge_ptr output;

public:
    project_orthogonal_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "project_orthogonal_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(message))
        {
            const auto &src = frame_msg->get_data();

            const auto dst = tensor<float, 3>::stack({src.max<1>({0}), src.max<1>({1}), src.max<1>({2})});

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

CEREAL_REGISTER_TYPE(project_orthogonal_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, project_orthogonal_node)

class soft_argmax_node : public graph_node
{
    float beta;
    graph_edge_ptr output;

public:
    soft_argmax_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    void set_beta(float value)
    {
        beta = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "soft_argmax_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(beta);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(message))
        {
            const auto &src = frame_msg->get_data();

            const auto &grid_msg = frame_msg->get_metadata<frame_message<tensor<float, 3>>>("grid");
            const auto &grid = grid_msg->get_data();

            const auto softmax_src = src.view<3>({src.shape[0] * src.shape[1], 0, src.shape[2], src.shape[3]}, {})
                .transform([this](const float value, auto...) {
                                return beta * value;
                            })
                .softmax(0);

            const auto dst = softmax_src.unsqueeze(0)
                .transform(grid.unsqueeze(2), [](const auto value1, const auto value2, auto...) {
                    return value1 * value2;
                })
                .sum<1>({1});

            const auto num = softmax_src.shape[1] * softmax_src.shape[2];

            const auto confs = softmax_src
                .unsqueeze().contiguous()
                .max<1>({0})
                .sum<2>({0, 1})
                .transform([num](const float value, auto...) {
                    return value / num;
                });

            auto new_obj_msg = std::make_shared<object_message>();

            const auto make_message = [frame_msg](const auto &data)
            {
                auto msg = std::make_shared<frame_message<std::decay_t<decltype(data)>>>();
                msg->set_data(std::move(data));
                msg->set_profile(frame_msg->get_profile());
                msg->set_timestamp(frame_msg->get_timestamp());
                msg->set_frame_number(frame_msg->get_frame_number());
                msg->set_metadata(*frame_msg);
                return msg;
            };

            new_obj_msg->add_field("pose_preds", make_message(dst));
            new_obj_msg->add_field("confs", make_message(confs));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(soft_argmax_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, soft_argmax_node)

class fusion_node : public graph_node
{
    graph_edge_ptr output;

public:
    fusion_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual std::string get_proc_name() const override
    {
        return "fusion_node";
    }

    static tensor<float, 2> fuse_pose_preds(const tensor<float, 3> &pose_preds, const tensor<float, 3> &weights)
    {
        const auto xy_weight = weights.view<2>({1, weights.shape[1], 0}, {0, 0, 0}).contiguous();
        const auto xz_weight = weights.view<2>({1, weights.shape[1], 0}, {0, 0, 1}).contiguous();
        const auto yz_weight = weights.view<2>({1, weights.shape[1], 0}, {0, 0, 2}).contiguous();
        const auto xy_pred = pose_preds.view<2>({2, pose_preds.shape[1], 0}, {0, 0, 0});
        const auto xz_pred = pose_preds.view<2>({2, pose_preds.shape[1], 0}, {0, 0, 1});
        const auto yz_pred = pose_preds.view<2>({2, pose_preds.shape[1], 0}, {0, 0, 2});

        const auto x_weight = tensor<float, 2>::concat<0>({xy_weight, xz_weight});
        const auto y_weight = tensor<float, 2>::concat<0>({xy_weight, yz_weight});
        const auto z_weight = tensor<float, 2>::concat<0>({xz_weight, yz_weight});

        const auto norm_x_weight = x_weight.view().transform(x_weight.sum<1>({0}).unsqueeze(0), [](const auto value1, const auto value2, auto...) {
            return value1 / value2;
        });
        const auto norm_y_weight = y_weight.view().transform(y_weight.sum<1>({0}).unsqueeze(0), [](const auto value1, const auto value2, auto...) {
            return value1 / value2;
        });
        const auto norm_z_weight = z_weight.view().transform(z_weight.sum<1>({0}).unsqueeze(0), [](const auto value1, const auto value2, auto...) {
            return value1 / value2;
        });

        const auto x_weight1 = norm_x_weight.view<2>({1, norm_x_weight.shape[1]}, {0, 0});
        const auto x_weight2 = norm_x_weight.view<2>({1, norm_x_weight.shape[1]}, {1, 0});
        const auto y_weight1 = norm_y_weight.view<2>({1, norm_y_weight.shape[1]}, {0, 0});
        const auto y_weight2 = norm_y_weight.view<2>({1, norm_y_weight.shape[1]}, {1, 0});
        const auto z_weight1 = norm_z_weight.view<2>({1, norm_z_weight.shape[1]}, {0, 0});
        const auto z_weight2 = norm_z_weight.view<2>({1, norm_z_weight.shape[1]}, {1, 0});

        const auto x_pred1 = xy_pred.view<2>({1, xy_pred.shape[1]}, {0, 0});
        const auto x_pred2 = xz_pred.view<2>({1, xz_pred.shape[1]}, {0, 0});
        const auto y_pred1 = xy_pred.view<2>({1, xy_pred.shape[1]}, {1, 0});
        const auto y_pred2 = yz_pred.view<2>({1, yz_pred.shape[1]}, {0, 0});
        const auto z_pred1 = xz_pred.view<2>({1, xz_pred.shape[1]}, {1, 0});
        const auto z_pred2 = yz_pred.view<2>({1, yz_pred.shape[1]}, {1, 0});

        const auto add = [](const auto value1, const auto value2, auto...) { return value1 + value2; };
        const auto mul = [](const auto value1, const auto value2, auto...) { return value1 * value2; };

        const auto x_pred = x_weight1
            .transform(x_pred1, mul)
            .transform(x_weight2.transform(x_pred2, mul), add);

        const auto y_pred = y_weight1
            .transform(y_pred1, mul)
            .transform(y_weight2.transform(y_pred2, mul), add);

        const auto z_pred = z_weight1
            .transform(z_pred1, mul)
            .transform(z_weight2.transform(z_pred2, mul), add);

        return tensor<float, 2>::concat<0>({x_pred, y_pred, z_pred});
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("pose_preds"));

            auto pose_preds = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(obj_msg->get_field("pose_preds"))->get_data();
            const auto &confs = std::dynamic_pointer_cast<frame_message<tensor<float, 1>>>(obj_msg->get_field("confs"))->get_data();
            const auto &offset = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("cubes"))->get_metadata<frame_message<std::array<float, 3>>>("offset")->get_data();
            const auto &weights = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(obj_msg->get_field("weights"))->get_data();

            const std::array<float, 2> offset_xy = {offset[0], offset[1]};
            pose_preds.view<2>({2, pose_preds.shape[1], 0}, {0, 0, 0}).assign([&offset_xy](const auto value, const auto i, auto...) {
                return value + offset_xy[i];
            });

            const std::array<float, 2> offset_xz = {offset[0], offset[2]};
            pose_preds.view<2>({2, pose_preds.shape[1], 0}, {0, 0, 1}).assign([&offset_xz](const auto value, const auto i, auto...) {
                return value + offset_xz[i];
            });

            const std::array<float, 2> offset_yz = {offset[1], offset[2]};
            pose_preds.view<2>({2, pose_preds.shape[1], 0}, {0, 0, 2}).assign([&offset_yz](const auto value, const auto i, auto...) {
                return value + offset_yz[i];
            });

            const auto fused_pose_preds = fuse_pose_preds(pose_preds, weights);

            {
                const auto num_points = fused_pose_preds.shape[1];

                std::ofstream ofs;
                ofs.open("./result.pcd", std::ios::out);

                ofs << "VERSION 0.7" << std::endl;
                ofs << "FIELDS x y z rgba" << std::endl;
                ofs << "SIZE 4 4 4 4" << std::endl;
                ofs << "TYPE F F F U" << std::endl;
                ofs << "COUNT 1 1 1 1" << std::endl;
                ofs << "WIDTH " << num_points << std::endl;
                ofs << "HEIGHT 1" << std::endl;
                ofs << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
                ofs << "POINTS " << num_points << std::endl;
                ofs << "DATA ascii" << std::endl;

                for (size_t j = 0; j < num_points; j++)
                {
                    ofs << fused_pose_preds.get({0, j}) << " " << fused_pose_preds.get({1, j}) << " " << fused_pose_preds.get({2, j}) << " " << 16711680 << std::endl;
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(fusion_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, fusion_node)

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

static void load_model(std::string model_path, std::vector<uint8_t> &data)
{
    std::ifstream ifs;
    ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
    if (ifs.fail())
    {
        std::cerr << "File open error: " << model_path << "\n";
        std::quick_exit(0);
    }

    ifs.seekg(0, std::ios::end);
    const auto length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    data.resize(length);

    ifs.read((char *)data.data(), length);
    if (ifs.fail())
    {
        std::cerr << "File read error: " << model_path << "\n";
        std::quick_exit(0);
    }
}

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
        const auto model_path = "../sample/faster-voxelpose/data/faster-voxelpose/backbone.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        backbone_model_data = std::move(data);
    }

    std::vector<uint8_t> pose_center_net_model_data;
    {
        const auto model_path = "../sample/faster-voxelpose/data/faster-voxelpose/pose_center_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        pose_center_net_model_data = std::move(data);
    }

    std::vector<uint8_t> pose_c2c_net_model_data;
    {
        const auto model_path = "../sample/faster-voxelpose/data/faster-voxelpose/pose_c2c_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        pose_c2c_net_model_data = std::move(data);
    }

    std::vector<uint8_t> joint_conv_net_model_data;
    {
        const auto model_path = "../sample/faster-voxelpose/data/faster-voxelpose/joint_conv_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        joint_conv_net_model_data = std::move(data);
    }

    std::vector<uint8_t> joint_weight_net_model_data;
    {
        const auto model_path = "../sample/faster-voxelpose/data/faster-voxelpose/joint_weight_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        joint_weight_net_model_data = std::move(data);
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

    const auto all_heatmaps = [&](){
        std::shared_ptr<frame_number_sync_node> sync(new frame_number_sync_node());
        for (const auto &[camera_name, heatmaps] : heatmaps_list)
        {
            sync->set_input(heatmaps, camera_name);
        }
        g->add_node(sync);

        return sync->get_output();
    }();

    std::shared_ptr<pre_project_whole_node> pre_project(new pre_project_whole_node());
    pre_project->set_input(all_heatmaps);
    pre_project->set_grid_center({0.0, -500.0, 800.0});
    g->add_node(pre_project);

    std::shared_ptr<project_whole_node> project(new project_whole_node());
    project->set_input(pre_project->get_output());
    project->set_grid_size({8000.0, 8000.0, 2000.0});
    project->set_cube_size({{80, 80, 20}});
    g->add_node(project);

    std::shared_ptr<onnx_runtime_node> inference_pose_center_net(new onnx_runtime_node());
    inference_pose_center_net->set_input(project->get_output());
    inference_pose_center_net->set_model_data(pose_center_net_model_data);
    g->add_node(inference_pose_center_net);

    const auto pose_centernet_output = [&]()
    {
        std::shared_ptr<frame_number_sync_node> sync(new frame_number_sync_node());
        sync->set_input(inference_pose_center_net->add_output("hm"), "hm");
        sync->set_input(inference_pose_center_net->add_output("size"), "size");
        sync->set_input(project->get_output(), "feats");
        g->add_node(sync);

        return sync->get_output();
    }();

    std::shared_ptr<center_decode_node> pose_center_decode(new center_decode_node());
    pose_center_decode->set_input(pose_centernet_output);
    pose_center_decode->set_max_num(10);
    g->add_node(pose_center_decode);

    std::shared_ptr<object_map_node> pose_center_decode_item(new object_map_node());
    pose_center_decode_item->set_input(pose_center_decode->get_output());
    g->add_node(pose_center_decode_item);

    std::shared_ptr<onnx_runtime_node> inference_pose_c2c_net(new onnx_runtime_node());
    inference_pose_c2c_net->set_input(pose_center_decode_item->add_output("topk_feats"));
    inference_pose_c2c_net->set_model_data(pose_c2c_net_model_data);
    g->add_node(inference_pose_c2c_net);

    const auto pose_centernet_proposal_inputs = [&]()
    {
        std::shared_ptr<frame_number_sync_node> sync(new frame_number_sync_node());
        sync->set_input(pose_center_decode_item->add_output("topk_2d_index"), "topk_2d_index");
        sync->set_input(pose_center_decode_item->add_output("topk_2d_confs"), "topk_2d_confs");
        sync->set_input(pose_center_decode_item->add_output("topk_size"), "topk_size");
        sync->set_input(inference_pose_c2c_net->add_output("output"), "feat_1d");
        g->add_node(sync);

        return sync->get_output();
    }();

    std::shared_ptr<proposal_node> proposal(new proposal_node());
    proposal->set_input(pose_centernet_proposal_inputs);
    proposal->set_max_num(10);
    proposal->set_threshold(0.3f);
    proposal->set_grid_size({8000.0, 8000.0, 2000.0});
    proposal->set_grid_center({0.0, -500.0, 800.0});
    proposal->set_cube_size({{80, 80, 20}});
    g->add_node(proposal);

    std::shared_ptr<object_map_node> pre_project_item(new object_map_node());
    pre_project_item->set_input(pre_project->get_output());
    g->add_node(pre_project_item);

    std::shared_ptr<frame_number_sync_node> heatmap_proposal_sync(new frame_number_sync_node());
    heatmap_proposal_sync->set_input(pre_project_item->add_output("heatmaps"), "heatmaps");
    heatmap_proposal_sync->set_input(proposal->get_output(), "proposal");
    g->add_node(heatmap_proposal_sync);

    std::shared_ptr<iterate_proposal_node> iterate_proposal(new iterate_proposal_node());
    iterate_proposal->set_input(heatmap_proposal_sync->get_output());
    g->add_node(iterate_proposal);

    std::shared_ptr<project_individual_node> project_proposal(new project_individual_node());
    project_proposal->set_input(iterate_proposal->get_output());
    project_proposal->set_grid_size({8000.0, 8000.0, 2000.0});
    project_proposal->set_grid_center({0.0, -500.0, 800.0});
    project_proposal->set_individual_grid_size({{2000.0, 2000.0, 2000.0}});
    project_proposal->set_cube_size({64, 64, 64});
    g->add_node(project_proposal);

    std::shared_ptr<project_orthogonal_node> project_ortho(new project_orthogonal_node());
    project_ortho->set_input(project_proposal->get_output());
    g->add_node(project_ortho);

    std::shared_ptr<onnx_runtime_node> inference_joint_conv_net(new onnx_runtime_node());
    inference_joint_conv_net->set_input(project_ortho->get_output());
    inference_joint_conv_net->set_model_data(joint_conv_net_model_data);
    g->add_node(inference_joint_conv_net);

    const auto joint_features = inference_joint_conv_net->add_output("output");

    std::shared_ptr<soft_argmax_node> soft_argmax(new soft_argmax_node());
    soft_argmax->set_input(joint_features);
    soft_argmax->set_beta(100);
    g->add_node(soft_argmax);

    std::shared_ptr<object_map_node> soft_argmax_item(new object_map_node());
    soft_argmax_item->set_input(soft_argmax->get_output());
    g->add_node(soft_argmax_item);

    std::shared_ptr<onnx_runtime_node> inference_joint_weight_net(new onnx_runtime_node());
    inference_joint_weight_net->set_input(joint_features);
    inference_joint_weight_net->set_model_data(joint_weight_net_model_data);
    g->add_node(inference_joint_weight_net);

    const auto fusion_input = [&]()
    {
        std::shared_ptr<frame_number_sync_node> sync(new frame_number_sync_node());
        sync->set_input(inference_joint_weight_net->add_output("output"), "weights");
        sync->set_input(soft_argmax_item->add_output("pose_preds"), "pose_preds");
        sync->set_input(soft_argmax_item->add_output("confs"), "confs");
        sync->set_input(project_proposal->get_output(), "cubes");
        g->add_node(sync);

        return sync->get_output();
    }();

    std::shared_ptr<fusion_node> fusion(new fusion_node());
    fusion->set_input(fusion_input);
    g->add_node(fusion);

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
