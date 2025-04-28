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

static cv::Mat get_affine_transform(const cv::Point2f &center, const cv::Size2f &scale, const cv::Size2f &output_size)
{
    const auto get_tri_3rd_point = [](const cv::Point2f &a, const cv::Point2f &b)
    {
        const auto direct = a - b;
        return b + cv::Point2f(-direct.y, direct.x);
    };

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

    cv::Mat result;
    cv::getAffineTransform(src_tri, dst_tri).convertTo(result, cv::DataType<float>::type);

    return result;
};

static cv::Mat get_transform(const cv::Point2f &center, const cv::Size2f &scale, const cv::Size2f &output_size)
{
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
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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
    std::shared_ptr<onnx_runtime_session> get_or_load(const std::vector<uint8_t> &model_data)
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

    static float median(std::vector<float> v)
    {
        using namespace std;

        const auto size = v.size();
        vector<float> _v(v.size());
        copy(v.begin(), v.end(), back_inserter(_v));
        float tmp;
        for (int i = 0; i < size - 1; i++)
        {
            for (int j = i + 1; j < size; j++)
            {
                if (_v[i] > _v[j])
                {
                    tmp = _v[i];
                    _v[i] = _v[j];
                    _v[j] = tmp;
                }
            }
        }
        if (size % 2 == 1)
        {
            return _v[(size - 1) / 2];
        }
        else
        {
            return (_v[(size / 2) - 1] + _v[size / 2]) / 2;
        }
    }

    static float mean(std::vector<float> v)
    {
        const auto size = v.size();
        float sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += v[i];
        }
        return sum / size;
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

                        const float *data = nullptr;
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

class pre_inference_node : public graph_node
{
    graph_edge_ptr output;

public:
    pre_inference_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "pre_inference_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }
    
    template <typename T, int32_t num_dims>
    static tensor<T, num_dims> make_tensor(std::initializer_list<T> values, const std::array<uint32_t, num_dims>& shape)
    {
        std::vector<T> vec(values);
        return tensor<T, num_dims>(shape, vec.data());
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            std::vector<tensor<float, 4>> images;
            std::vector<tensor<float, 1>> cameras_fx;
            std::vector<tensor<float, 1>> cameras_fy;
            std::vector<tensor<float, 1>> cameras_cx;
            std::vector<tensor<float, 1>> cameras_cy;
            std::vector<tensor<float, 3>> cameras_rotation;
            std::vector<tensor<float, 3>> cameras_translation;
            std::vector<tensor<float, 3>> cameras_standard_translation;
            std::vector<tensor<float, 3>> cameras_k;
            std::vector<tensor<float, 3>> cameras_p;
            std::vector<tensor<float, 2>> images_center;
            std::vector<tensor<float, 2>> images_scale;
            std::vector<tensor<float, 2>> images_rotation;
            std::vector<tensor<float, 3>> images_transform;
            std::vector<camera_data> cameras;
            std::vector<roi_data> rois;

            auto dot = [](const std::array<double, 3> &a, const std::array<double, 3> &b)
            {
                return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            };

            std::shared_ptr<frame_message_base> base_frame_msg = nullptr;
            for (const auto &[name, field] : obj_msg->get_fields())
            {
                if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(field))
                {
                    auto &src = frame_msg->get_data();

                    const auto camera_msg = frame_msg->get_metadata<camera_data_message>("camera");
                    const auto camera = camera_msg->get_data();

                    const auto roi_msg = frame_msg->get_metadata<roi_data_message>("roi");
                    const auto roi = roi_msg->get_data();

                    images.emplace_back(src);
                    cameras.emplace_back(camera);
                    rois.emplace_back(roi);

                    const std::array<double, 3> standard_translation = {
                        -dot(camera.rotation[0], camera.translation),
                        -dot(camera.rotation[1], camera.translation),
                        -dot(camera.rotation[2], camera.translation),
                    };
                    
                    cameras_fx.push_back(make_tensor<float, 1>({static_cast<float>(camera.fx)}, {1}));
                    cameras_fy.push_back(make_tensor<float, 1>({static_cast<float>(camera.fy)}, {1}));
                    cameras_cx.push_back(make_tensor<float, 1>({static_cast<float>(camera.cx)}, {1}));
                    cameras_cy.push_back(make_tensor<float, 1>({static_cast<float>(camera.cy)}, {1}));
                    cameras_rotation.push_back(make_tensor<float, 3>({static_cast<float>(camera.rotation[0][0]),
                                                                      static_cast<float>(camera.rotation[0][1]),
                                                                      static_cast<float>(camera.rotation[0][2]),
                                                                      static_cast<float>(camera.rotation[1][0]),
                                                                      static_cast<float>(camera.rotation[1][1]),
                                                                      static_cast<float>(camera.rotation[1][2]),
                                                                      static_cast<float>(camera.rotation[2][0]),
                                                                      static_cast<float>(camera.rotation[2][1]),
                                                                      static_cast<float>(camera.rotation[2][2])},
                                                                     {1, 3, 3}));
                    cameras_translation.push_back(make_tensor<float, 3>({static_cast<float>(camera.translation[0]),
                                                                         static_cast<float>(camera.translation[1]),
                                                                         static_cast<float>(camera.translation[2])},
                                                                        {1, 3, 1}));
                    cameras_standard_translation.push_back(make_tensor<float, 3>({static_cast<float>(standard_translation[0]),
                                                                                  static_cast<float>(standard_translation[1]),
                                                                                  static_cast<float>(standard_translation[2])},
                                                                                 {1, 3, 1}));
                    cameras_k.push_back(make_tensor<float, 3>({static_cast<float>(camera.k[0]),
                                                               static_cast<float>(camera.k[1]),
                                                               static_cast<float>(camera.k[2])},
                                                              {1, 3, 1}));
                    cameras_p.push_back(make_tensor<float, 3>({static_cast<float>(camera.p[0]),
                                                               static_cast<float>(camera.p[1])},
                                                              {1, 2, 1}));

                    images_center.push_back(make_tensor<float, 2>({static_cast<float>(roi.center[0]),
                                                                   static_cast<float>(roi.center[1])},
                                                                  {1, 2}));
                    images_scale.push_back(make_tensor<float, 2>({static_cast<float>(roi.scale[0]),
                                                                  static_cast<float>(roi.scale[1])},
                                                                 {1, 2}));
                    images_rotation.push_back(make_tensor<float, 2>({static_cast<float>(roi.rotation)},
                                                                    {1, 1}));

                    const auto image_center = cv::Point2f(static_cast<float>(roi.center[0]), static_cast<float>(roi.center[1]));
                    const auto image_scale = cv::Size2f(static_cast<float>(roi.scale[0]), static_cast<float>(roi.scale[1]));
                    const auto &&image_size = cv::Size2f(960, 512);

                    const auto image_trans = get_affine_transform(image_center,
                                                                  image_scale,
                                                                  image_size);

                    images_transform.push_back(make_tensor<float, 3>({image_trans.at<float>(0, 0),
                                                                      image_trans.at<float>(0, 1),
                                                                      image_trans.at<float>(0, 2),
                                                                      image_trans.at<float>(1, 0),
                                                                      image_trans.at<float>(1, 1),
                                                                      image_trans.at<float>(1, 2),
                                                                      0, 0, 1},
                                                                     {1, 3, 3}));

                    base_frame_msg = frame_msg;
                }
            }

            if (base_frame_msg == nullptr)
            {
                return;
            }

            const auto views_image = tensor<float, 4>::concat<3>(images);
            const auto image_center = tensor<float, 2>::stack(images_center);
            const auto image_scale = tensor<float, 2>::stack(images_scale);
            const auto image_rotation = tensor<float, 2>::stack(images_rotation);
            const auto image_transform = tensor<float, 3>::stack(images_transform);
            const auto camera_fx = tensor<float, 1>::stack(cameras_fx);
            const auto camera_fy = tensor<float, 1>::stack(cameras_fy);
            const auto camera_cx = tensor<float, 1>::stack(cameras_cx);
            const auto camera_cy = tensor<float, 1>::stack(cameras_cy);
            const auto camera_rotation = tensor<float, 3>::stack(cameras_rotation);
            const auto camera_translation = tensor<float, 3>::stack(cameras_translation);
            const auto camera_standard_translation = tensor<float, 3>::stack(cameras_standard_translation);
            const auto camera_k = tensor<float, 3>::stack(cameras_k);
            const auto camera_p = tensor<float, 3>::stack(cameras_p);

            auto new_obj_msg = std::make_shared<object_message>();

            const auto make_message = [base_frame_msg](const auto &output_tensor)
            {
                auto msg = std::make_shared<frame_message<std::decay_t<decltype(output_tensor)>>>();
                msg->set_data(std::move(output_tensor));
                msg->set_profile(base_frame_msg->get_profile());
                msg->set_timestamp(base_frame_msg->get_timestamp());
                msg->set_frame_number(base_frame_msg->get_frame_number());
                msg->set_metadata(*base_frame_msg);
                return msg;
            };

            new_obj_msg->add_field("views", make_message(views_image));
            new_obj_msg->add_field("image_center", make_message(image_center));
            new_obj_msg->add_field("image_scale", make_message(image_scale));
            new_obj_msg->add_field("image_rotation", make_message(image_rotation));
            new_obj_msg->add_field("image_trans", make_message(image_transform));
            new_obj_msg->add_field("camera_fx", make_message(camera_fx));
            new_obj_msg->add_field("camera_fy", make_message(camera_fy));
            new_obj_msg->add_field("camera_cx", make_message(camera_cx));
            new_obj_msg->add_field("camera_cy", make_message(camera_cy));
            new_obj_msg->add_field("camera_k", make_message(camera_k));
            new_obj_msg->add_field("camera_p", make_message(camera_p));
            new_obj_msg->add_field("camera_R", make_message(camera_rotation));
            new_obj_msg->add_field("camera_T", make_message(camera_translation));
            new_obj_msg->add_field("camera_standard_T", make_message(camera_standard_translation));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(pre_inference_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, pre_inference_node)

class post_inference_node : public graph_node
{
    graph_edge_ptr output;

    std::array<float, 3> grid_center;
    std::array<float, 3> grid_size;

public:
    post_inference_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "post_inference_node";
    }

    std::array<float, 3> get_grid_center() const
    {
        return grid_center;
    }
    void set_grid_center(const std::array<float, 3> &value)
    {
        grid_center = value;
    }
    std::array<float, 3> get_grid_size() const
    {
        return grid_size;
    }
    void set_grid_size(const std::array<float, 3> &value)
    {
        grid_size = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(grid_center, grid_size);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto &pred_logits = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(obj_msg->get_field("pred_logits"))->get_data();
            const auto &pred_poses = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(obj_msg->get_field("pred_poses"))->get_data();

            constexpr float threshold = 0.1f;
            constexpr uint32_t num_joints = 15;
            constexpr uint32_t num_queries = 10;

            auto norm2absolute = [](const std::array<float, 3>& coord, const std::array<float, 3>& grid_center, const std::array<float, 3>& grid_size)
            {
                std::array<float, 3> result;
                for (size_t i = 0; i < 3; i++)
                {
                    result[i] = coord[i] * grid_size[i] + grid_center[i] - grid_size[i] / 2.0f;
                }
                return result;
            };

            assert(num_queries == pred_logits.shape[1]);
            assert(3 == pred_poses.shape[0]);
            assert(num_queries * num_joints == pred_poses.shape[1]);

            std::vector<float> scores;
            std::vector<std::vector<std::array<float, 3>>> poses;
            for (uint32_t i = 0; i < num_queries; i++)
            {
                const auto score = 1.0f / (1.0f + exp(-pred_logits.get({1, i, 0})));
                const auto det = (score > threshold) - 1.0f;

                if (det >= 0.0f)
                {
                    std::vector<std::array<float, 3>> joints;

                    for (uint32_t j = 0; j < num_joints; j++)
                    {
                        std::array<float, 3> coord;
                        for (uint32_t k = 0; k < 3; k++)
                        {
                            coord[k] = pred_poses.get({k, i * num_joints + j, 0});
                        }
                        const auto joint = norm2absolute(coord, grid_center, grid_size);
                        joints.push_back(joint);
                    }

                    scores.push_back(score);
                    poses.push_back(joints);
                }
            }

            const auto num_points = poses.size() * num_joints;

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

            for (size_t i = 0; i < poses.size(); i++)
            {
                const auto& joints = poses[i];

                for (size_t j = 0; j < joints.size(); j++)
                {
                    const auto& joint = joints[j];
                    ofs << joint[0] << " " << joint[1] << " " << joint[2] << " " << 16711680 << std::endl;
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(post_inference_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, post_inference_node)

class local_server
{
    asio::io_context io_context;
    std::shared_ptr<graph_proc_server> server;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;

public:
    local_server()
        : io_context(), server(std::make_shared<graph_proc_server>(io_context, "0.0.0.0", 31400)), th(), running(false)
    {
    }

    void run()
    {
        running = true;
        th.reset(new std::thread([this]
                                 { io_context.run(); }));
    }

    void stop()
    {
        if (running.load())
        {
            running.store(false);
            io_context.stop();
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

    asio::io_context io_context;

    std::shared_ptr<subgraph> g(new subgraph());

    std::vector<std::tuple<int32_t, int32_t>> camera_list = {
        {0, 12},
        {0, 6},
        {0, 23},
        {0, 13},
        {0, 3},
    };

    std::shared_ptr<panoptic_data_loader_node> data_loader(new panoptic_data_loader_node());
    data_loader->set_data_dir("/workspace/panoptic-toolbox/data");
    data_loader->set_sequence_list({"171204_pose1"});
    data_loader->set_camera_list(camera_list);
    g->add_node(data_loader);

    std::shared_ptr<object_map_node> map_data(new object_map_node());
    map_data->set_input(data_loader->get_output());
    g->add_node(map_data);

    std::unordered_map<std::string, graph_edge_ptr> images;
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

        images[camera_name] = normalized;
    }

    std::shared_ptr<frame_number_sync_node> sync(new frame_number_sync_node());
    for (const auto &[camera_name, image] : images)
    {
        sync->set_input(image, camera_name);
    }
    g->add_node(sync);

    std::shared_ptr<pre_inference_node> pre_inference(new pre_inference_node());
    pre_inference->set_input(sync->get_output());
    g->add_node(pre_inference);

    std::vector<uint8_t> transformer_model_data;
    {
        const auto model_path = "../sample/mvp/data/mvp/model.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        transformer_model_data = std::move(data);
    }

    std::shared_ptr<onnx_runtime_node> inference_transformer(new onnx_runtime_node());
    inference_transformer->set_input(pre_inference->get_output());
    inference_transformer->set_model_data(transformer_model_data);
    g->add_node(inference_transformer);

    const auto pred_logits = inference_transformer->add_output("pred_logits");
    const auto pred_poses = inference_transformer->add_output("pred_poses");

    std::shared_ptr<frame_number_sync_node> sync_outputs(new frame_number_sync_node());
    sync_outputs->set_input(pred_logits, "pred_logits");
    sync_outputs->set_input(pred_poses, "pred_poses");
    g->add_node(sync_outputs);

    std::shared_ptr<post_inference_node> post_inference(new post_inference_node());
    post_inference->set_input(sync_outputs->get_output());
    post_inference->set_grid_size({8000.0, 8000.0, 2000.0});
    post_inference->set_grid_center({0.0, -500.0, 800.0});
    g->add_node(post_inference);

    graph_proc_client client;
    client.deploy(io_context, "127.0.0.1", 31400, g);

    on_shutdown_handlers.push_back([&client, &server]
                                   {
        client.stop();
        server.stop(); });

    std::thread io_thread([&io_context]
                          { io_context.run(); });

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
