#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <tuple>
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
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

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

class image_loader_node : public heartbeat_node
{
    std::string filename;
    bool flip;
    graph_edge_ptr output;

    uint64_t frame_number;

public:
    image_loader_node()
        : heartbeat_node(), filename(), flip(false), output(std::make_shared<graph_edge>(this)), frame_number(0)
    {
        set_output(output);
    }

    void set_filename(std::string value)
    {
        filename = value;
    }
    const std::string &get_filename() const
    {
        return filename;
    }
    void set_flip(bool value)
    {
        flip = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "image_loader_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(cereal::base_class<heartbeat_node>(this));
        archive(filename, flip);
    }

    virtual void tick() override
    {
        auto data = cv::imread(filename, cv::IMREAD_UNCHANGED | cv::IMREAD_IGNORE_ORIENTATION);

        cv::Mat input_img;
        cv::cvtColor(data, input_img, cv::COLOR_BGR2RGB);

        if (flip)
        {
            cv::flip(input_img, input_img, 1);
        }

        tensor<uint8_t, 4> input_img_tensor({static_cast<std::uint32_t>(input_img.size().width),
                                             static_cast<std::uint32_t>(input_img.size().height),
                                             static_cast<std::uint32_t>(input_img.elemSize()),
                                             1},
                                            (const uint8_t *)input_img.data,
                                            {static_cast<std::uint32_t>(input_img.step[1]),
                                             static_cast<std::uint32_t>(input_img.step[0]),
                                             static_cast<std::uint32_t>(1),
                                             static_cast<std::uint32_t>(input_img.total())});

        auto frame_msg = std::make_shared<frame_message<tensor<float, 4>>>();

        const auto input_img_tensor_f = input_img_tensor.cast<float>().transform([this](const float value, const size_t w, const size_t h, const size_t c, const size_t n)
                                                                                 { return value / 255.0f; });

        frame_msg->set_data(std::move(input_img_tensor_f));
        frame_msg->set_timestamp(0);
        frame_msg->set_frame_number(frame_number++);

        output->send(frame_msg);
    }
};

CEREAL_REGISTER_TYPE(image_loader_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, image_loader_node)

class parameter_loader_node : public graph_node
{
    std::string filename;
    graph_edge_ptr output;

public:
    parameter_loader_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    void set_filename(std::string value)
    {
        filename = value;
    }

    const std::string &get_filename() const
    {
        return filename;
    }

    virtual std::string get_proc_name() const override
    {
        return "parameter_loader_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(filename);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(message))
        {
            std::ifstream file(filename.c_str());
            nlohmann::json params = nlohmann::json::parse(file);

            auto obj_msg = std::make_shared<object_message>();

            for (const auto& item : params.items())
            {
                const auto &name = item.key();
                const auto &value = item.value();

                const auto type = value["type"].get<std::string>();
                assert(type == "float");
                const auto shape = value["shape"].get<std::vector<uint32_t>>();
                const auto data = value["data"].get<std::vector<float>>();

                graph_message_ptr output_msg;

                if (shape.size() == 2)
                {
                    constexpr auto num_dims = 2;

                    auto msg = std::make_shared<frame_message<tensor<float, num_dims>>>();
                    tensor<float, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(1)),
                                                        static_cast<std::uint32_t>(shape.at(0))},
                                                        data.data());

                    msg->set_data(std::move(output_tensor));
                    msg->set_profile(frame_msg->get_profile());
                    msg->set_timestamp(frame_msg->get_timestamp());
                    msg->set_frame_number(frame_msg->get_frame_number());
                    msg->set_metadata(*frame_msg);

                    obj_msg->add_field(name, msg);
                }
            }

            output->send(obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(parameter_loader_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, parameter_loader_node)

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

static auto l2norm(auto a, auto b, auto c)
{
    constexpr auto eps = 1e-12f;
    return std::max({std::sqrt(a * a + b * b + c * c), eps});
};

static auto dot(auto a0, auto a1, auto a2, auto b0, auto b1, auto b2)
{
    return a0 * b0 + a1 * b1 + a2 * b2;
};

static auto cross(auto a0, auto a1, auto a2, auto b0, auto b1, auto b2)
{
    return std::make_tuple(a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0);
};

static auto normalize(auto a, auto b, auto c)
{
    const auto norm = l2norm(a, b, c);
    return std::make_tuple(a / norm, b / norm, c / norm);
};

template <typename Tensor>
static tensor<float, 4> rot6d_to_rotmat(const Tensor &src)
{
    tensor<float, 4> dst({3, 3, src.shape[1], src.shape[2]});

    for (uint32_t i = 0; i < src.shape[2]; i++)
    {
        for (uint32_t j = 0; j < src.shape[1]; j++)
        {
            const auto x0 = src.get({0, j, i});
            const auto x1 = src.get({2, j, i});
            const auto x2 = src.get({4, j, i});
            const auto x3 = src.get({1, j, i});
            const auto x4 = src.get({3, j, i});
            const auto x5 = src.get({5, j, i});

            const auto [y0, y1, y2] = normalize(x0, x1, x2);
            const auto num = dot(y0, y1, y2, x3, x4, x5);
            const auto [y3, y4, y5] = normalize(x3 - num * y0, x4 - num * y1, x5 - num * y2);
            const auto [y6, y7, y8] = cross(y0, y1, y2, y3, y4, y5);

            dst.set({0, 0, j, i}, y0);
            dst.set({0, 1, j, i}, y1);
            dst.set({0, 2, j, i}, y2);
            dst.set({1, 0, j, i}, y3);
            dst.set({1, 1, j, i}, y4);
            dst.set({1, 2, j, i}, y5);
            dst.set({2, 0, j, i}, y6);
            dst.set({2, 1, j, i}, y7);
            dst.set({2, 2, j, i}, y8);
        }
    }

    return dst;
}

template <typename Tensor>
static tensor<float, 4> flip_rotmat(const Tensor &src)
{
    tensor<float, 4> dst(src.shape);

    for (uint32_t i = 0; i < src.shape[3]; i++)
    {
        for (uint32_t j = 0; j < src.shape[2]; j++)
        {
            dst.set({0, 0, j, i}, src.get({0, 0, j, i}));
            dst.set({1, 0, j, i}, src.get({1, 0, j, i}) * -1);
            dst.set({2, 0, j, i}, src.get({2, 0, j, i}) * -1);
            dst.set({0, 1, j, i}, src.get({0, 1, j, i}) * -1);
            dst.set({1, 1, j, i}, src.get({1, 1, j, i}));
            dst.set({2, 1, j, i}, src.get({2, 1, j, i}));
            dst.set({0, 2, j, i}, src.get({0, 2, j, i}) * -1);
            dst.set({1, 2, j, i}, src.get({1, 2, j, i}));
            dst.set({2, 2, j, i}, src.get({2, 2, j, i}));
        }
    }

    return dst;
}

static tensor<float, 2> normalize(const tensor<float, 2> &src)
{
    assert(src.shape.size() == 2);

    tensor<float, 2> dst(src.shape);

    for (uint32_t i = 0; i < src.shape[1]; i++)
    {
        float acc = 0.0f;
        for (uint32_t j = 0; j < src.shape[0]; j++)
        {
            acc += std::pow(src.get({j, i}), 2.0f);
        }
        const auto denom = std::max(std::pow(acc, 1.0f / 2.0f), 1e-12f);
        for (uint32_t j = 0; j < src.shape[0]; j++)
        {
            dst.set({j, i}, src.get({j, i}) / denom);
        }
    }

    return dst;
}

class prepare_body_mesh_parameter_node : public graph_node
{
    graph_edge_ptr output;

public:
    prepare_body_mesh_parameter_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "prepare_body_mesh_parameter_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto base_frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("pred_pose"));

            const auto &pred_pose = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_pose"))->get_data();
            const auto &pred_shape = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_shape"))->get_data();
            const auto &pred_rhand = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_rhand"))->get_data();
            const auto &pred_lhand = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_lhand"))->get_data();
            const auto &pred_face = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_face"))->get_data();
            const auto &pred_exp = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_exp"))->get_data();

            const auto hand_rotmat_mean = rot6d_to_rotmat(pred_rhand.view<3>({6, pred_rhand.shape[0] / 6, pred_rhand.shape[1]}));
            const auto left_hand_pose = flip_rotmat(hand_rotmat_mean);
            const auto right_hand_pose = hand_rotmat_mean;

            const auto pred_face_rotmat = rot6d_to_rotmat(pred_face.view<3>({6, pred_face.shape[0] / 6, pred_face.shape[1]}));
            const auto jaw_pose = pred_face_rotmat.view<4>({pred_face_rotmat.shape[0], pred_face_rotmat.shape[1], 1, pred_face_rotmat.shape[3]},
                                                           {0, 0, 0, 0}).contiguous();
            const auto leye_pose = pred_face_rotmat.view<4>({pred_face_rotmat.shape[0], pred_face_rotmat.shape[1], 1, pred_face_rotmat.shape[3]},
                                                            {0, 0, 1, 0}).contiguous();
            const auto reye_pose = pred_face_rotmat.view<4>({pred_face_rotmat.shape[0], pred_face_rotmat.shape[1], 1, pred_face_rotmat.shape[3]},
                                                            {0, 0, 2, 0}).contiguous();
            const auto expression = pred_exp;
            const auto pred_rotmat_body = rot6d_to_rotmat(pred_pose.view<3>({6, pred_pose.shape[0] / 6, pred_pose.shape[1]}));
            const auto body_pose = pred_rotmat_body.view<4>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 21, pred_rotmat_body.shape[3]},
                                                            {0, 0, 1, 0}).contiguous();
            const auto global_orient = pred_rotmat_body.view<4>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 1, pred_rotmat_body.shape[3]},
                                                                {0, 0, 0, 0}).contiguous();

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

            new_obj_msg->add_field("beta", make_message(pred_shape));
            new_obj_msg->add_field("global_orient", make_message(global_orient));
            new_obj_msg->add_field("body_pose", make_message(body_pose));
            new_obj_msg->add_field("left_hand_pose", make_message(left_hand_pose));
            new_obj_msg->add_field("right_hand_pose", make_message(right_hand_pose));
            new_obj_msg->add_field("jaw_pose", make_message(jaw_pose));
            new_obj_msg->add_field("leye_pose", make_message(leye_pose));
            new_obj_msg->add_field("reye_pose", make_message(reye_pose));
            new_obj_msg->add_field("expression", make_message(expression));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(prepare_body_mesh_parameter_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, prepare_body_mesh_parameter_node)

class prepare_face_mesh_parameter_node : public graph_node
{
    graph_edge_ptr output;

public:
    prepare_face_mesh_parameter_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "prepare_face_mesh_parameter_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto base_frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("pred_pose"));

            const auto &pred_pose = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_pose"))->get_data();
            const auto &pred_shape = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_shape"))->get_data();
            const auto &pred_orient = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_orient"))->get_data();
            const auto &pred_exp = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_exp"))->get_data();

            const auto pred_pose_rotmat = rot6d_to_rotmat(pred_pose.view<3>({6, pred_pose.shape[0] / 6, pred_pose.shape[1]}));
            const auto jaw_pose = pred_pose_rotmat.view<4>({pred_pose_rotmat.shape[0], pred_pose_rotmat.shape[1], 1, pred_pose_rotmat.shape[3]},
                                                           {0, 0, 0, 0})
                                      .contiguous();
            const auto leye_pose = pred_pose_rotmat.view<4>({pred_pose_rotmat.shape[0], pred_pose_rotmat.shape[1], 1, pred_pose_rotmat.shape[3]},
                                                            {0, 0, 1, 0})
                                       .contiguous();
            const auto reye_pose = pred_pose_rotmat.view<4>({pred_pose_rotmat.shape[0], pred_pose_rotmat.shape[1], 1, pred_pose_rotmat.shape[3]},
                                                            {0, 0, 2, 0})
                                       .contiguous();
            const auto pred_orient_rotmat = rot6d_to_rotmat(pred_orient.view<3>({6, pred_orient.shape[0] / 6, pred_orient.shape[1]}));
            const auto global_orient = pred_orient_rotmat.view<4>({pred_orient_rotmat.shape[0], pred_orient_rotmat.shape[1], 1, pred_orient_rotmat.shape[3]},
                                                                  {0, 0, 0, 0})
                                           .contiguous();

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

            new_obj_msg->add_field("beta", make_message(pred_shape));
            new_obj_msg->add_field("global_orient", make_message(global_orient));
            new_obj_msg->add_field("jaw_pose", make_message(jaw_pose));
            new_obj_msg->add_field("leye_pose", make_message(leye_pose));
            new_obj_msg->add_field("reye_pose", make_message(reye_pose));
            new_obj_msg->add_field("expression", make_message(pred_exp));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(prepare_face_mesh_parameter_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, prepare_face_mesh_parameter_node)

class prepare_hand_mesh_parameter_node : public graph_node
{
    graph_edge_ptr output;

public:
    prepare_hand_mesh_parameter_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "prepare_hand_mesh_parameter_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto base_frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("pred_pose"));

            const auto &pred_pose = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_pose"))->get_data();
            const auto &pred_shape = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_shape"))->get_data();
            const auto &pred_orient = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_orient"))->get_data();

            const auto pred_pose_rotmat = rot6d_to_rotmat(pred_pose.view<3>({6, pred_pose.shape[0] / 6, pred_pose.shape[1]}));
            const auto hand_pose = pred_pose_rotmat.view<4>({pred_pose_rotmat.shape[0], pred_pose_rotmat.shape[1], pred_pose_rotmat.shape[2], pred_pose_rotmat.shape[3]},
                                                            {0, 0, 0, 0})
                                       .contiguous();
            const auto pred_orient_rotmat = rot6d_to_rotmat(pred_orient.view<3>({6, pred_orient.shape[0] / 6, pred_orient.shape[1]}));
            const auto global_orient = pred_orient_rotmat.view<4>({pred_orient_rotmat.shape[0], pred_orient_rotmat.shape[1], 1, pred_orient_rotmat.shape[3]},
                                                                  {0, 0, 0, 0})
                                           .contiguous();

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

            new_obj_msg->add_field("beta", make_message(pred_shape));
            new_obj_msg->add_field("global_orient", make_message(global_orient));
            new_obj_msg->add_field("hand_pose", make_message(hand_pose));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(prepare_hand_mesh_parameter_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, prepare_hand_mesh_parameter_node)

class optimize_body_mesh_parameter_node : public graph_node
{
    std::string filename;
    graph_edge_ptr output;

    std::shared_ptr<object_message> mesh_spec;

public:
    optimize_body_mesh_parameter_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    void set_filename(std::string value)
    {
        filename = value;
    }

    const std::string &get_filename() const
    {
        return filename;
    }

    virtual std::string get_proc_name() const override
    {
        return "optimize_body_mesh_parameter_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(filename);
    }

    static std::shared_ptr<object_message> load_parameters(const std::string &filename, std::shared_ptr<frame_message_base> frame_msg = nullptr)
    {
        std::ifstream file(filename.c_str());
        nlohmann::json params = nlohmann::json::parse(file);

        auto obj_msg = std::make_shared<object_message>();

        for (const auto &item : params.items())
        {
            const auto &name = item.key();
            const auto &value = item.value();

            const auto type = value["type"].get<std::string>();
            const auto shape = value["shape"].get<std::vector<uint32_t>>();

            graph_message_ptr output_msg;

            if (type == "float")
            {
                using elem_type = float;
                const auto data = value["data"].get<std::vector<elem_type>>();
                if (shape.size() == 2)
                {
                    constexpr auto num_dims = 2;

                    auto msg = std::make_shared<frame_message<tensor<elem_type, num_dims>>>();
                    tensor<elem_type, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(1)),
                                                               static_cast<std::uint32_t>(shape.at(0))},
                                                              data.data());

                    msg->set_data(std::move(output_tensor));
                    if (frame_msg)
                    {
                        msg->set_profile(frame_msg->get_profile());
                        msg->set_timestamp(frame_msg->get_timestamp());
                        msg->set_frame_number(frame_msg->get_frame_number());
                        msg->set_metadata(*frame_msg);
                    }

                    obj_msg->add_field(name, msg);
                }
                if (shape.size() == 3)
                {
                    constexpr auto num_dims = 3;

                    auto msg = std::make_shared<frame_message<tensor<elem_type, num_dims>>>();
                    tensor<elem_type, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(2)),
                                                               static_cast<std::uint32_t>(shape.at(1)),
                                                               static_cast<std::uint32_t>(shape.at(0))},
                                                              data.data());

                    msg->set_data(std::move(output_tensor));
                    if (frame_msg)
                    {
                        msg->set_profile(frame_msg->get_profile());
                        msg->set_timestamp(frame_msg->get_timestamp());
                        msg->set_frame_number(frame_msg->get_frame_number());
                        msg->set_metadata(*frame_msg);
                    }

                    obj_msg->add_field(name, msg);
                }
            }
            else if (type == "int")
            {
                using elem_type = std::int32_t;
                const auto data = value["data"].get<std::vector<elem_type>>();
                if (shape.size() == 1)
                {
                    constexpr auto num_dims = 1;

                    auto msg = std::make_shared<frame_message<tensor<elem_type, num_dims>>>();
                    tensor<elem_type, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(0))},
                                                              data.data());

                    msg->set_data(std::move(output_tensor));
                    if (frame_msg)
                    {
                        msg->set_profile(frame_msg->get_profile());
                        msg->set_timestamp(frame_msg->get_timestamp());
                        msg->set_frame_number(frame_msg->get_frame_number());
                        msg->set_metadata(*frame_msg);
                    }

                    obj_msg->add_field(name, msg);
                }
                if (shape.size() == 2)
                {
                    constexpr auto num_dims = 2;

                    auto msg = std::make_shared<frame_message<tensor<elem_type, num_dims>>>();
                    tensor<elem_type, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(1)),
                                                               static_cast<std::uint32_t>(shape.at(0))},
                                                              data.data());

                    msg->set_data(std::move(output_tensor));
                    if (frame_msg)
                    {
                        msg->set_profile(frame_msg->get_profile());
                        msg->set_timestamp(frame_msg->get_timestamp());
                        msg->set_frame_number(frame_msg->get_frame_number());
                        msg->set_metadata(*frame_msg);
                    }

                    obj_msg->add_field(name, msg);
                }
            }
        }

        return obj_msg;
    }

    virtual void initialize() override
    {
        mesh_spec = load_parameters(filename);
    }

    tensor<float, 4> get_global_rotation(const tensor<float, 4> &global_orient, const tensor<float, 4> &pose) const
    {
        const auto &tpose_joints = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(mesh_spec->get_field("tpose_joints"))->get_data();
        const auto &parents = std::dynamic_pointer_cast<frame_message<tensor<int32_t, 1>>>(mesh_spec->get_field("parents"))->get_data();

        const auto rel_joints = tpose_joints.view({tpose_joints.shape[0], 21, tpose_joints.shape[2]}, {0, 1, 0})
            .transform([&](const float value, const size_t i, const size_t j, auto...) {
                return value - tpose_joints.get({i, parents.get({j + 1})});
            });

        tensor<float, 4> transforms({4, 4, pose.shape[2] + 1, pose.shape[3]});
        for (uint32_t i = 0; i < transforms.shape[3]; i++)
        {
            {
                for (uint32_t k = 0; k < 3; k++)
                {
                    for (uint32_t l = 0; l < 3; l++)
                    {
                        transforms.set({l, k, 0, i}, global_orient.get({l, k, 0, i}));
                    }
                    transforms.set({3, k, 0, i}, 0);
                }
                for (uint32_t k = 0; k < 3; k++)
                {
                    transforms.set({3, k, 0, i}, tpose_joints.get({k, 0, 0}));
                }
                transforms.set({3, 3, 0, i}, 1);
            }
            for (uint32_t j = 1; j < transforms.shape[2]; j++)
            {
                for (uint32_t k = 0; k < 3; k++)
                {
                    for (uint32_t l = 0; l < 3; l++)
                    {
                        transforms.set({l, k, j, i}, pose.get({l, k, j - 1, i}));
                    }
                    transforms.set({3, k, j, i}, 0);
                }
                for (uint32_t k = 0; k < 3; k++)
                {
                    transforms.set({3, k, j, i}, rel_joints.get({k, j - 1, 0}));
                }
                transforms.set({3, 3, j, i}, 1);
            }
        }
        for (uint32_t i = 0; i < transforms.shape[3]; i++)
        {
            for (uint32_t j = 1; j < transforms.shape[2]; j++)
            {
                auto abs_transform = tensor<float, 2>::zeros({4, 4});
                for (uint32_t k = 0; k < 4; k++)
                {
                    for (uint32_t l = 0; l < 4; l++)
                    {
                        for (uint32_t m = 0; m < 4; m++)
                        {
                            abs_transform.set({l, k}, abs_transform.get({l, k}) + transforms.get({m, k, parents.get({j}), i}) * transforms.get({l, m, j, i}));
                        }
                    }
                }
                transforms.view<2>({4, 4, 0, 0}, {0, 0, j, i}).assign(abs_transform.view(), [](auto, const float value, auto...) { return value; });
            }
        }

        return transforms.view<4>({3, 3, transforms.shape[2], transforms.shape[3]}, {0, 0, 0, 0}).contiguous();
    }

    static tensor<float, 3> bmm(const tensor<float, 3>& value1, const tensor<float, 3>& value2)
    {
        assert(value1.shape[0] == value2.shape[1]);
        auto result = tensor<float, 3>::zeros({value1.shape[1], value2.shape[0], value1.shape[2]});
        for (uint32_t i = 0; i < value1.shape[2]; i++)
        {
            for (uint32_t m = 0; m < result.shape[1]; m++)
            {
                for (uint32_t n = 0; n < result.shape[0]; n++)
                {
                    for (uint32_t k = 0; k < value1.shape[0]; k++)
                    {
                        result.set({n, m, i}, result.get({n, m, i}) + value1.get({k, m, i}) * value2.get({n, k, i}));
                    }
                }
            }
        }
        return result;
    }

    static tensor<float, 3> blend_shapes(const tensor<float, 2> &betas, const tensor<float, 3> &shape_disps)
    {
        auto result = tensor<float, 3>::zeros({shape_disps.shape[1], shape_disps.shape[2], betas.shape[1]});
        for (uint32_t b = 0; b < betas.shape[1]; b++)
        {
            for (uint32_t l = 0; l < betas.shape[0]; l++)
            {
                for (uint32_t m = 0; m < shape_disps.shape[2]; m++)
                {
                    for (uint32_t k = 0; k < shape_disps.shape[1]; k++)
                    {
                        result.set({k, m, b}, result.get({k, m, b}) + betas.get({l, b}) * shape_disps.get({l, k, m}));
                    }
                }
            }
        }
        return result;
    }

    tensor<float, 3> get_tpose(const tensor<float, 2>& betas) const
    {
        const auto &joint_template = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(mesh_spec->get_field("joint_template"))->get_data();
        const auto &joint_dirs = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(mesh_spec->get_field("joint_dirs"))->get_data();

        const auto joints = joint_template.view<3>({joint_template.shape[0], joint_template.shape[1], 1})
                                .transform(blend_shapes(betas, joint_dirs).view(),
                                           [](const float value1, const float value2, auto...)
                                           { return value1 + value2; });
        return joints;
    }

    static inline float sign(float x)
    {
        return (x >= 0.0f) ? +1.0f : -1.0f;
    }

    static inline float norm(float a, float b, float c, float d)
    {
        return sqrt(a * a + b * b + c * c + d * d);
    }

    template <size_t size>
    static inline float norm(const std::array<float, size> &values)
    {
        float result = 0.0f;
        for (size_t i = 0; i < size; i++)
        {
            result += values[i] * values[i];
        }
        return std::sqrt(result);
    }

    template <size_t size>
    static inline float sum(const std::array<float, size> &values)
    {
        float result = 0.0f;
        for (size_t i = 0; i < size; i++)
        {
            result += values[i];
        }
        return result;
    }

    template <size_t size>
    static inline std::array<float, size> normalize(const std::array<float, size> &values)
    {
        const auto n = norm(values);
        std::array<float, size> result;
        for (size_t i = 0; i < size; i++)
        {
            result[i] = values[i] / n;
        }
        return result;
    }

    template <size_t size>
    static inline float dot(const std::array<float, size> &values1, const std::array<float, size> &values2)
    {
        float result = 0.0f;
        for (size_t i = 0; i < size; i++)
        {
            result += values1[i] * values2[i];
        }
        return result;
    }

    template <size_t size>
    static inline std::array<float, size> mul(const std::array<float, size> &v, const float x)
    {
        std::array<float, size> result;
        for (size_t i = 0; i < size; i++)
        {
            result[i] = v[i] * x;
        }
        return result;
    }

    static std::array<std::array<float, 3>, 3> quaternion_to_rotation_matrix(const std::array<float, 4>& quaternion)
    {
        const auto q0 = quaternion[0];
        const auto q1 = quaternion[1];
        const auto q2 = quaternion[2];
        const auto q3 = quaternion[3];

        const auto r00 = 2 * (q0 * q0 + q1 * q1) - 1;
        const auto r01 = 2 * (q1 * q2 - q0 * q3);
        const auto r02 = 2 * (q1 * q3 + q0 * q2);

        const auto r10 = 2 * (q1 * q2 + q0 * q3);
        const auto r11 = 2 * (q0 * q0 + q2 * q2) - 1;
        const auto r12 = 2 * (q2 * q3 - q0 * q1);

        const auto r20 = 2 * (q1 * q3 - q0 * q2);
        const auto r21 = 2 * (q2 * q3 + q0 * q1);
        const auto r22 = 2 * (q0 * q0 + q3 * q3) - 1;

        return {std::array<float, 3>{r00, r01, r02},
                std::array<float, 3>{r10, r11, r12},
                std::array<float, 3>{r20, r21, r22}};
    }

    static std::array<float, 4> rotation_matrix_to_quaternion(const std::array<std::array<float, 3>, 3>& rotation_matrix, float eps = 1e-6f)
    {
        float r11 = rotation_matrix[0][0];
        float r12 = rotation_matrix[0][1];
        float r13 = rotation_matrix[0][2];
        float r21 = rotation_matrix[1][0];
        float r22 = rotation_matrix[1][1];
        float r23 = rotation_matrix[1][2];
        float r31 = rotation_matrix[2][0];
        float r32 = rotation_matrix[2][1];
        float r33 = rotation_matrix[2][2];
        float q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
        float q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
        float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
        float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
        if (q0 < 0.0f)
        {
            q0 = 0.0f;
        }
        if (q1 < 0.0f)
        {
            q1 = 0.0f;
        }
        if (q2 < 0.0f)
        {
            q2 = 0.0f;
        }
        if (q3 < 0.0f)
        {
            q3 = 0.0f;
        }
        q0 = sqrt(q0);
        q1 = sqrt(q1);
        q2 = sqrt(q2);
        q3 = sqrt(q3);
        if (q0 >= q1 && q0 >= q2 && q0 >= q3)
        {
            q0 *= +1.0f;
            q1 *= sign(r32 - r23);
            q2 *= sign(r13 - r31);
            q3 *= sign(r21 - r12);
        }
        else if (q1 >= q0 && q1 >= q2 && q1 >= q3)
        {
            q0 *= sign(r32 - r23);
            q1 *= +1.0f;
            q2 *= sign(r21 + r12);
            q3 *= sign(r13 + r31);
        }
        else if (q2 >= q0 && q2 >= q1 && q2 >= q3)
        {
            q0 *= sign(r13 - r31);
            q1 *= sign(r21 + r12);
            q2 *= +1.0f;
            q3 *= sign(r32 + r23);
        }
        else if (q3 >= q0 && q3 >= q1 && q3 >= q2)
        {
            q0 *= sign(r21 - r12);
            q1 *= sign(r31 + r13);
            q2 *= sign(r32 + r23);
            q3 *= +1.0f;
        }
        else
        {
            return {};
        }
        float r = norm(q0, q1, q2, q3);
        q0 /= r;
        q1 /= r;
        q2 /= r;
        q3 /= r;
        
        return {q0, q1, q2, q3};
    }

    static std::array<float, 3> quaternion_to_angle_axis(const std::array<float, 4>& quaternion)
    {
        const auto q1 = quaternion[1];
        const auto q2 = quaternion[2];
        const auto q3 = quaternion[3];
        const auto sin_theta = std::hypot(q1, q2, q3);

        std::array<float, 3> angle_axis;

        if (std::fpclassify(sin_theta) != FP_ZERO)
        {
            const auto cos_theta = quaternion[0];
            const auto two_theta =
                2.0f * ((cos_theta < 0.0f) ? std::atan2(-sin_theta, -cos_theta)
                                               : std::atan2(sin_theta, cos_theta));
            const auto k = two_theta / sin_theta;
            angle_axis[0] = q1 * k;
            angle_axis[1] = q2 * k;
            angle_axis[2] = q3 * k;
        }
        else
        {
            const auto k = 2.0f;
            angle_axis[0] = q1 * k;
            angle_axis[1] = q2 * k;
            angle_axis[2] = q3 * k;
        }

        return angle_axis;
    }

    static std::tuple<std::array<std::array<float, 3>, 3>, float> compute_twist_rotation(const std::array<std::array<float, 3>, 3> &rotation_matrix, const std::array<float, 3> &twist_axis)
    {
        const auto quaternion = rotation_matrix_to_quaternion(rotation_matrix);
        
        const auto norm_twist_axis = normalize(twist_axis);
        const auto projection = mul(norm_twist_axis, dot(norm_twist_axis, {quaternion[1], quaternion[2], quaternion[3]}));

        const auto twist_quaternion = normalize<4>({quaternion[0], projection[0], projection[1], projection[2]});

        const auto twist_rotation = quaternion_to_rotation_matrix(twist_quaternion);
        const auto twist_aa = quaternion_to_angle_axis(twist_quaternion);
        const auto twist_angle = sum(twist_aa) / sum(norm_twist_axis);
        return std::make_tuple(twist_rotation, twist_angle);
    }

    static std::tuple<tensor<float, 3>, tensor<float, 1>> compute_twist_rotation(const tensor<float, 3>& rotation_matrix, const tensor<float, 2>& twist_axis)
    {
        assert(rotation_matrix.shape[2] == twist_axis.shape[1]);
        const auto batch_size = rotation_matrix.shape[2];
        tensor<float, 3> twist_rotation({3, 3, batch_size});
        tensor<float, 1> twist_angle({batch_size});

        for (uint32_t i = 0; i < batch_size; i++)
        {
            std::array<std::array<float, 3>, 3> rotation_matrix_block;
            for (uint32_t m = 0; m < 3; m++)
            {
                for (uint32_t n = 0; n < 3; n++)
                {
                    rotation_matrix_block[m][n] = rotation_matrix.get({n, m, i});
                }
            }

            std::array<float, 3> twist_axis_block;
            for (uint32_t m = 0; m < 3; m++)
            {
                twist_axis_block[m] = twist_axis.get({m, i});
            }

            const auto [twist_rotation_block, twist_angle_block] = compute_twist_rotation(rotation_matrix_block, twist_axis_block);

            for (uint32_t m = 0; m < 3; m++)
            {
                for (uint32_t n = 0; n < 3; n++)
                {
                    twist_rotation.set({n, m, i}, twist_rotation_block[m][n]);
                }
            }
            twist_angle.set({i}, twist_angle_block);
        }

        return std::make_tuple(twist_rotation, twist_angle);
    }

    template <size_t size_m, size_t size_n, size_t size_k>
    static std::array<std::array<float, 3>, 3> mul(const std::array<std::array<float, size_k>, size_m> &value1, const std::array<std::array<float, size_n>, size_k> &value2)
    {
        std::array<std::array<float, size_n>, size_m> result = {};
        for (size_t m = 0; m < size_m; m++)
        {
            for (size_t n = 0; n < size_n; n++)
            {
                for (size_t k = 0; k < size_k; k++)
                {
                    result[m][n] += value1[m][k] * value2[k][n];
                }
            }
        }
        return result;
    }

    static tensor<float, 3> batch_rodrigues(const tensor<float, 2>& rot_vec)
    {
        const auto batch_size = rot_vec.shape[1];

        tensor<float, 3> result({3, 3, batch_size});

        for (uint32_t i = 0; i < batch_size; i++)
        {
            const auto angle = norm<3>({rot_vec.get({0, i}) + 1e-8f, rot_vec.get({1, i}) + 1e-8f, rot_vec.get({2, i}) + 1e-8f});

            const auto rx = rot_vec.get({0, i}) / angle;
            const auto ry = rot_vec.get({1, i}) / angle;
            const auto rz = rot_vec.get({2, i}) / angle;

            std::array<std::array<float, 3>, 3> k = {{
                {0, -rz, ry},
                {rz, 0, -rx},
                {-ry, rx, 0}
            }};

            std::array<std::array<float, 3>, 3> kk = mul(k, k);

            std::array<std::array<float, 3>, 3> rot_mat;
            for (size_t m = 0; m < 3; m++)
            {
                for (size_t n = 0; n < 3; n++)
                {
                    rot_mat[m][n] = (m == n) + std::sin(angle) * k[m][n] + (1 - std::cos(angle)) * kk[m][n];
                }
            }

            for (size_t m = 0; m < 3; m++)
            {
                for (size_t n = 0; n < 3; n++)
                {
                    result.set({n, m, i}, rot_mat[m][n]);
                }
            }
        }

        return result;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto base_frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("pred_pose"));

            const auto &pred_pose = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_pose"))->get_data();
            const auto &pred_shape = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_shape"))->get_data();
            const auto &pred_rhand = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_rhand"))->get_data();
            const auto &pred_lhand = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_lhand"))->get_data();
            const auto &pred_face = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_face"))->get_data();
            const auto &pred_exp = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_exp"))->get_data();
            const auto &pred_orient_lhand = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_orient_lhand"))->get_data();
            const auto &pred_orient_rhand = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("pred_orient_rhand"))->get_data();

            const auto hand_rotmat_mean = rot6d_to_rotmat(pred_rhand.view<3>({6, pred_rhand.shape[0] / 6, pred_rhand.shape[1]}));
            const auto left_hand_pose = flip_rotmat(hand_rotmat_mean);
            const auto right_hand_pose = hand_rotmat_mean;

            const auto pred_face_rotmat = rot6d_to_rotmat(pred_face.view<3>({6, pred_face.shape[0] / 6, pred_face.shape[1]}));
            const auto jaw_pose = pred_face_rotmat.view<4>({pred_face_rotmat.shape[0], pred_face_rotmat.shape[1], 1, pred_face_rotmat.shape[3]},
                                                           {0, 0, 0, 0}).contiguous();
            const auto leye_pose = pred_face_rotmat.view<4>({pred_face_rotmat.shape[0], pred_face_rotmat.shape[1], 1, pred_face_rotmat.shape[3]},
                                                            {0, 0, 1, 0}).contiguous();
            const auto reye_pose = pred_face_rotmat.view<4>({pred_face_rotmat.shape[0], pred_face_rotmat.shape[1], 1, pred_face_rotmat.shape[3]},
                                                            {0, 0, 2, 0}).contiguous();
            const auto expression = pred_exp;

            auto pred_rotmat_body = rot6d_to_rotmat(pred_pose.view<3>({6, pred_pose.shape[0] / 6, pred_pose.shape[1]}));

            {
                const auto body_pose = pred_rotmat_body.view<4>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 21, pred_rotmat_body.shape[3]},
                                                                {0, 0, 1, 0})
                                           .contiguous();
                const auto global_orient = pred_rotmat_body.view<4>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 1, pred_rotmat_body.shape[3]},
                                                                    {0, 0, 0, 0})
                                               .contiguous();

                const auto body_global_pose = get_global_rotation(global_orient, body_pose);

                const auto pred_global_lelbow = body_global_pose.view<3>({3, 3, 1, 0}, {0, 0, 18, 0}).transpose({1, 0, 2}).contiguous();
                const auto pred_global_relbow = body_global_pose.view<3>({3, 3, 1, 0}, {0, 0, 19, 0}).transpose({1, 0, 2}).contiguous();

                const auto target_global_lwrist = flip_rotmat(rot6d_to_rotmat(pred_orient_lhand.view<3>({6, pred_orient_lhand.shape[0] / 6, pred_orient_lhand.shape[1]}))).view<3>({3, 3, 1, 0}, {0, 0, 0, 0}).contiguous();
                const auto target_global_rwrist = rot6d_to_rotmat(pred_orient_rhand.view<3>({6, pred_orient_rhand.shape[0] / 6, pred_orient_rhand.shape[1]})).view<3>({3, 3, 1, 0}, {0, 0, 0, 0}).contiguous();

                auto opt_lwrist = bmm(pred_global_lelbow, target_global_lwrist);
                auto opt_rwrist = bmm(pred_global_relbow, target_global_rwrist);

                const auto tpose_joints = get_tpose(pred_shape);

                const auto lelbow_twist_axis = ::normalize(tpose_joints.view<2>({tpose_joints.shape[0], 0, tpose_joints.shape[2]}, {0, 20, 0})
                                                .transform(tpose_joints.view<2>({tpose_joints.shape[0], 0, tpose_joints.shape[2]}, {0, 18, 0}),
                                                            [](const float value1, const float value2, auto...)
                                                            { return value1 - value2; }));

                const auto relbow_twist_axis = ::normalize(tpose_joints.view<2>({tpose_joints.shape[0], 0, tpose_joints.shape[2]}, {0, 21, 0})
                                                .transform(tpose_joints.view<2>({tpose_joints.shape[0], 0, tpose_joints.shape[2]}, {0, 19, 0}),
                                                            [](const float value1, const float value2, auto...)
                                                            { return value1 - value2; }));

                const auto lelbow_twist_angle = std::get<1>(compute_twist_rotation(opt_lwrist, lelbow_twist_axis));
                const auto relbow_twist_angle = std::get<1>(compute_twist_rotation(opt_rwrist, relbow_twist_axis));
                const auto pi = 3.141592653589793f;
                const auto min_angle = -0.4f * pi;
                const auto max_angle = 0.4f * pi;

                const auto norm_lelbow_twist_angle = lelbow_twist_angle.transform([&](const float value, auto...)
                                                                                {
                    if (value == std::clamp(value, min_angle, max_angle))
                    {
                        return 0.0f;
                    }
                    else if (value > max_angle)
                    {
                        return value - max_angle;
                    }
                    else if (value < min_angle)
                    {
                        return value - min_angle;
                    }
                    return value; });

                const auto norm_relbow_twist_angle = relbow_twist_angle.transform([&](const float value, auto...)
                                                                                {
                    if (value == std::clamp(value, min_angle, max_angle))
                    {
                        return 0.0f;
                    }
                    else if (value > max_angle)
                    {
                        return value - max_angle;
                    }
                    else if (value < min_angle)
                    {
                        return value - min_angle;
                    }
                    return value; });

                const auto lelbow_twist_angle_axis = lelbow_twist_axis.transform([&](const float value, const size_t i, const size_t j)
                                                                                { return value * norm_lelbow_twist_angle.get({j}); });

                const auto relbow_twist_angle_axis = relbow_twist_axis.transform([&](const float value, const size_t i, const size_t j)
                                                                                { return value * norm_relbow_twist_angle.get({j}); });

                const auto lelbow_twist = batch_rodrigues(lelbow_twist_angle_axis);
                const auto relbow_twist = batch_rodrigues(relbow_twist_angle_axis);

                const std::vector<float> vis_lhand = {0.3012627f};
                const std::vector<float> vis_rhand = {0.41886702f};
                const std::vector<float> vis_face = {0.7983043f};

                const auto hand_vis_th = 0.1f;
                const auto head_vis_th = 0.5f;

                opt_lwrist = bmm(lelbow_twist.transpose({1, 0, 2}).contiguous(), opt_lwrist);
                opt_rwrist = bmm(relbow_twist.transpose({1, 0, 2}).contiguous(), opt_rwrist);

                auto opt_lelbow = bmm(pred_rotmat_body.view<3>({3, 3, 1, 0}, {0, 0, 18, 0}).contiguous(), lelbow_twist);
                auto opt_relbow = bmm(pred_rotmat_body.view<3>({3, 3, 1, 0}, {0, 0, 19, 0}).contiguous(), relbow_twist);

                opt_lwrist.view().assign(pred_rotmat_body.view<3>({3, 3, 1, 0}, {0, 0, 20, 0}), [&](const float value1, const float value2, auto, auto, const size_t i)
                                        { return (vis_lhand[i] > hand_vis_th) ? value1 : value2; });
                opt_rwrist.view().assign(pred_rotmat_body.view<3>({3, 3, 1, 0}, {0, 0, 21, 0}), [&](const float value1, const float value2, auto, auto, const size_t i)
                                        { return (vis_rhand[i] > hand_vis_th) ? value1 : value2; });
                opt_lelbow.view().assign(pred_rotmat_body.view<3>({3, 3, 1, 0}, {0, 0, 18, 0}), [&](const float value1, const float value2, auto, auto, const size_t i)
                                        { return (vis_lhand[i] > hand_vis_th) ? value1 : value2; });
                opt_relbow.view().assign(pred_rotmat_body.view<3>({3, 3, 1, 0}, {0, 0, 19, 0}), [&](const float value1, const float value2, auto, auto, const size_t i)
                                        { return (vis_rhand[i] > hand_vis_th) ? value1 : value2; });

                pred_rotmat_body.view<3>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 0, pred_rotmat_body.shape[3]}, {0, 0, 18, 0}).assign(opt_lelbow.view());
                pred_rotmat_body.view<3>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 0, pred_rotmat_body.shape[3]}, {0, 0, 19, 0}).assign(opt_relbow.view());
                pred_rotmat_body.view<3>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 0, pred_rotmat_body.shape[3]}, {0, 0, 20, 0}).assign(opt_lwrist.view());
                pred_rotmat_body.view<3>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 0, pred_rotmat_body.shape[3]}, {0, 0, 21, 0}).assign(opt_rwrist.view());
            }

            const auto body_pose = pred_rotmat_body.view<4>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 21, pred_rotmat_body.shape[3]},
                                                            {0, 0, 1, 0})
                                       .contiguous();
            const auto global_orient = pred_rotmat_body.view<4>({pred_rotmat_body.shape[0], pred_rotmat_body.shape[1], 1, pred_rotmat_body.shape[3]},
                                                                {0, 0, 0, 0})
                                           .contiguous();

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

            new_obj_msg->add_field("beta", make_message(pred_shape));
            new_obj_msg->add_field("global_orient", make_message(global_orient));
            new_obj_msg->add_field("body_pose", make_message(body_pose));
            new_obj_msg->add_field("left_hand_pose", make_message(left_hand_pose));
            new_obj_msg->add_field("right_hand_pose", make_message(right_hand_pose));
            new_obj_msg->add_field("jaw_pose", make_message(jaw_pose));
            new_obj_msg->add_field("leye_pose", make_message(leye_pose));
            new_obj_msg->add_field("reye_pose", make_message(reye_pose));
            new_obj_msg->add_field("expression", make_message(expression));

            output->send(new_obj_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(optimize_body_mesh_parameter_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, optimize_body_mesh_parameter_node)

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

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            for (const auto &[name, output] : get_outputs())
            {
                try
                {
                    const auto &field = obj_msg->get_field(name);
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

class render_node : public graph_node
{
    graph_edge_ptr output;

public:
    render_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "render_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
        }
    }
};

CEREAL_REGISTER_TYPE(render_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, render_node)

class project_joint_iwp_node : public graph_node
{
    graph_edge_ptr output;

public:
    project_joint_iwp_node()
        : graph_node(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "project_joint_iwp_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    template <typename TensorNx3, typename Tensor3x3, typename Tensor3, typename Scalar>
    static TensorNx3 perspective_projection(const TensorNx3 &points, const Tensor3x3 &rotation, const Tensor3 &translation, const Scalar focal_length, const Tensor3 &camera_center)
    {
        assert(points.shape.size() == 3);

        TensorNx3 dst({2, points.shape[1], points.shape[2]});

        for (uint32_t i = 0; i < points.shape[2]; i++)
        {
            std::array<std::array<Scalar, 4>, 3> trans = {};
            for (uint32_t m = 0; m < 3; m++)
            {
                for (uint32_t n = 0; n < 3; n++)
                {
                    trans[m][n] = rotation.get({n, m, i});
                }
                trans[m][3] = translation.get({m, i});
            }

            std::array<std::array<Scalar, 3>, 2> cam = {};
            cam[0][0] = focal_length;
            cam[1][1] = focal_length;
            cam[2][2] = 1;
            cam[0][2] = camera_center.get({0, i});
            cam[1][2] = camera_center.get({1, i});

            for (uint32_t j = 0; j < points.shape[1]; j++)
            {
                std::array<Scalar, 4> pt;
                for (uint32_t k = 0; k < 3; k++)
                {
                    pt[k] = points.get({k, j, i});
                }
                pt[3] = Scalar(1);

                for (uint32_t m = 0; m < 3; m++)
                {
                    auto acc = Scalar(0);
                    for (uint32_t n = 0; n < 4; n++)
                    {
                        acc += pt[n] * trans[m][n];
                    }
                    pt[m] = acc;
                }
                for (uint32_t m = 0; m < 3; m++)
                {
                    pt[m] /= pt[2];
                }
                for (uint32_t m = 0; m < 2; m++)
                {
                    auto acc = Scalar(0);
                    for (uint32_t n = 0; n < 3; n++)
                    {
                        acc += pt[n] * cam[m][n];
                    }
                    pt[m] = acc;
                }

                dst.set({0, j, i}, pt[0]);
                dst.set({1, j, i}, pt[1]);
            }
        }

        return dst;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            const auto base_frame_msg = std::dynamic_pointer_cast<frame_message_base>(obj_msg->get_field("joints"));

            const auto &joints = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(obj_msg->get_field("joints"))->get_data();
            const auto &cam_sxy = std::dynamic_pointer_cast<frame_message<tensor<float, 2>>>(obj_msg->get_field("cam_sxy"))->get_data();

            const auto batch_size = joints.shape[2];
            auto rotation = tensor<float, 3>::zeros({3, 3, batch_size});
            auto translation = tensor<float, 2>::zeros({3, batch_size});
            for (uint32_t i = 0; i < batch_size; i++)
            {
                for (uint32_t k = 0; k < 3; k++)
                {
                    rotation.set({k, k, i}, 1);
                }
                translation.set({0, i}, cam_sxy.get({1, i}));
                translation.set({1, i}, cam_sxy.get({2, i}));
                translation.set({2, i}, 2.f * 5000.f / (224.f * cam_sxy.get({0, i}) + 1e-9f));
            }
            const auto camera_center = tensor<float, 2>::zeros({2, batch_size});

            auto proj = perspective_projection(joints, rotation, translation, 5000.0f, camera_center);
            proj.view().assign([](const auto value, auto...) { return value / (224. / 2.); });

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

            const auto new_msg = make_message(proj);

            output->send(new_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(project_joint_iwp_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, project_joint_iwp_node)

class extract_point_2d_node : public graph_node
{
    graph_edge_ptr output;
    uint32_t index;

public:
    extract_point_2d_node()
        : graph_node(), output(std::make_shared<graph_edge>(this)), index(0)
    {
        set_output(output);
    }

    void set_index(uint32_t value)
    {
        index = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "extract_point_2d_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(index);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 3>>>(message))
        {
            const auto &joints = frame_msg->get_data();

            const auto batch_size = joints.shape[2];
            tensor<float, 2> dst({2, batch_size});
            for (uint32_t i = 0; i < batch_size; i++)
            {
                dst.set({0, i}, joints.get({0, index, i}));
                dst.set({1, i}, joints.get({1, index, i}));
            }

            const auto make_message = [frame_msg](const auto &output_tensor)
            {
                auto msg = std::make_shared<frame_message<std::decay_t<decltype(output_tensor)>>>();
                msg->set_data(std::move(output_tensor));
                msg->set_profile(frame_msg->get_profile());
                msg->set_timestamp(frame_msg->get_timestamp());
                msg->set_frame_number(frame_msg->get_frame_number());
                msg->set_metadata(*frame_msg);
                return msg;
            };

            const auto new_msg = make_message(dst);

            output->send(new_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(extract_point_2d_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, extract_point_2d_node)

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

    ifs.read((char*)data.data(), length);
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

    std::unordered_map<std::string, std::vector<uint8_t>> encoder_model_data;

    std::unordered_map<std::string, std::string> encode_model_files = {
        {"body", "../sample/pymafx/data/body_encoder.onnx"},
        {"lhand", "../sample/pymafx/data/hand_encoder.onnx"},
        {"rhand", "../sample/pymafx/data/hand_encoder.onnx"},
        {"face", "../sample/pymafx/data/face_encoder.onnx"},
    };

    for (const auto &[part, model_path] : encode_model_files)
    {
        std::vector<uint8_t> data;
        load_model(model_path, data);

        encoder_model_data.emplace(part, std::move(data));
    }

    std::unordered_map<std::string, std::string> part_image_files = {
        {"body", "../sample/pymafx/data/img_body.png"},
        {"face", "../sample/pymafx/data/img_face.png"},
        {"lhand", "../sample/pymafx/data/img_lhand.png"},
        {"rhand", "../sample/pymafx/data/img_rhand.png"},
    };

    constexpr auto num_iter = 3;

    graph_edge_ptr features_ = nullptr;
    std::unordered_map<std::string, std::vector<graph_edge_ptr>> part_features;

    for (const auto& [part, part_img] : part_image_files)
    {
        std::shared_ptr<image_loader_node> data_loader(new image_loader_node());
        data_loader->set_filename(part_img);
        data_loader->set_interval(100000);
        data_loader->set_flip(part == "lhand");
        g->add_node(data_loader);

        std::shared_ptr<normalize_node> normalize(new normalize_node());
        normalize->set_input(data_loader->get_output());
        normalize->set_mean({0.485, 0.456, 0.406});
        normalize->set_std({0.229, 0.224, 0.225});
        g->add_node(normalize);

        std::shared_ptr<onnx_runtime_node> inference_encoder(new onnx_runtime_node());
        inference_encoder->set_input(normalize->get_output());
        inference_encoder->set_model_data(encoder_model_data[part]);
        g->add_node(inference_encoder);

        std::vector<graph_edge_ptr> iter_features;
        for (size_t i = 0; i < num_iter; i++)
        {
            const auto features = inference_encoder->add_output(fmt::format("output{}", i));
            iter_features.push_back(features);

            features_ = features;
        }
        part_features.emplace(part, iter_features);
    }

    std::shared_ptr<parameter_loader_node> init_mesh_parameter_loader(new parameter_loader_node());
    init_mesh_parameter_loader->set_input(features_);
    init_mesh_parameter_loader->set_filename("../sample/pymafx/data/init_mesh_parameter.json");
    g->add_node(init_mesh_parameter_loader);

    std::shared_ptr<object_map_node> initialize_mesh_parameter_obj(new object_map_node());
    initialize_mesh_parameter_obj->set_input(init_mesh_parameter_loader->get_output());
    g->add_node(initialize_mesh_parameter_obj);

    // iter 0

#if 1
    graph_edge_ptr pred_pose_body, pred_shape_body, pred_cam_body, pred_rotmat_body;
    {
        std::vector<uint8_t> body_grid_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_grid_feature_encoder0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            body_grid_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_hmr0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_body_grid_feature(new onnx_runtime_node());
        inference_body_grid_feature->set_input(part_features["body"][0]);
        inference_body_grid_feature->set_model_data(body_grid_feature_model_data);
        g->add_node(inference_body_grid_feature);

        std::shared_ptr<frame_number_sync_node> sync1(new frame_number_sync_node());
        {
            sync1->set_input(inference_body_grid_feature->add_output("output"), "x");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_pose"), "pred_pose.1");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_shape"), "pred_shape.1");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_cam"), "pred_cam.1");
        }
        g->add_node(sync1);

        std::shared_ptr<onnx_runtime_node> inference_hmr0(new onnx_runtime_node());
        inference_hmr0->set_input(sync1->get_output());
        inference_hmr0->set_model_data(hmr_model_data);
        g->add_node(inference_hmr0);

        pred_pose_body = inference_hmr0->add_output("pred_pose");
        pred_shape_body = inference_hmr0->add_output("pred_shape");
        pred_cam_body = inference_hmr0->add_output("pred_cam");
        pred_rotmat_body = inference_hmr0->add_output("pred_rotmat");
    }
#endif

#if 1
    graph_edge_ptr pred_pose_face, pred_exp_face, pred_orient_face, pred_shape_face, pred_cam_face;
    {
        std::vector<uint8_t> face_grid_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_grid_feature_encoder0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            face_grid_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> face_hmr0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_hmr0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            face_hmr0_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_face_grid_feature(new onnx_runtime_node());
        inference_face_grid_feature->set_input(part_features["face"][0]);
        inference_face_grid_feature->set_model_data(face_grid_feature_model_data);
        g->add_node(inference_face_grid_feature);

        std::shared_ptr<frame_number_sync_node> sync1(new frame_number_sync_node());
        {
            sync1->set_input(inference_face_grid_feature->add_output("output"), "x");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_face"), "pred_face.1");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_exp"), "pred_exp.1");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_shape_face"), "pred_shape.1");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_orient"), "pred_orient.1");
            sync1->set_input(initialize_mesh_parameter_obj->add_output("init_cam"), "pred_cam.1");
        }
        g->add_node(sync1);

        std::shared_ptr<onnx_runtime_node> inference_face_hmr0(new onnx_runtime_node());
        inference_face_hmr0->set_input(sync1->get_output());
        inference_face_hmr0->set_model_data(face_hmr0_model_data);
        g->add_node(inference_face_hmr0);

        pred_pose_face = inference_face_hmr0->add_output("pred_face");
        pred_exp_face = inference_face_hmr0->add_output("pred_exp");
        pred_shape_face = inference_face_hmr0->add_output("pred_shape");
        pred_orient_face = inference_face_hmr0->add_output("pred_orient");
        pred_cam_face = inference_face_hmr0->add_output("pred_cam");
    }
#endif

#if 1
    std::unordered_map<std::string, graph_edge_ptr> pred_pose_hands, pred_shape_hands, pred_orient_hands, pred_cam_hands;
    for (const auto& part : {"lhand", "rhand"})
    {
        std::vector<uint8_t> hand_grid_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_grid_feature_encoder0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hand_grid_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hand_hmr0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_hmr0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hand_hmr0_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_hand_grid_feature(new onnx_runtime_node());
        inference_hand_grid_feature->set_input(part_features[part][0]);
        inference_hand_grid_feature->set_model_data(hand_grid_feature_model_data);
        g->add_node(inference_hand_grid_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_hand_grid_feature->add_output("output"), "x");
            sync_hmr_input->set_input(initialize_mesh_parameter_obj->add_output(fmt::format("init_{}", part)), "pred_hand.1");
            sync_hmr_input->set_input(initialize_mesh_parameter_obj->add_output(fmt::format("init_shape_{}", part)), "pred_shape.1");
            sync_hmr_input->set_input(initialize_mesh_parameter_obj->add_output("init_orient"), "pred_orient.1");
            sync_hmr_input->set_input(initialize_mesh_parameter_obj->add_output("init_cam"), "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hand_hmr0_model_data);
        g->add_node(inference_hmr);

        const auto pred_pose_hand = inference_hmr->add_output("pred_hand");
        const auto pred_shape_hand = inference_hmr->add_output("pred_shape");
        const auto pred_orient_hand = inference_hmr->add_output("pred_orient");
        const auto pred_cam_hand = inference_hmr->add_output("pred_cam");

        pred_pose_hands.emplace(part, pred_pose_hand);
        pred_shape_hands.emplace(part, pred_shape_hand);
        pred_orient_hands.emplace(part, pred_orient_hand);
        pred_cam_hands.emplace(part, pred_cam_hand);
    }
#endif

#if 1
    graph_edge_ptr body_vertices, face_joints;
    std::unordered_map<std::string, graph_edge_ptr> hands_joints;
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_body, "pred_pose");
            sync_mesh_input->set_input(pred_shape_body, "pred_shape");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_rhand"), "pred_rhand");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_lhand"), "pred_lhand");
            sync_mesh_input->set_input(pred_pose_face, "pred_face");
            sync_mesh_input->set_input(pred_exp_face, "pred_exp");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<prepare_body_mesh_parameter_node> prepare_mesh_parameter(new prepare_body_mesh_parameter_node());
        prepare_mesh_parameter->set_input(sync_mesh_input->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        body_vertices = inference_mesh->add_output("smpl_vertices");
        hands_joints.emplace("lhand", inference_mesh->add_output("lhand_joints"));
        hands_joints.emplace("rhand", inference_mesh->add_output("rhand_joints"));
        face_joints = inference_mesh->add_output("face_joints");
    }
#endif

#if 1
    graph_edge_ptr face_vertices;
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_face, "pred_pose");
            sync_mesh_input->set_input(pred_shape_face, "pred_shape");
            sync_mesh_input->set_input(pred_orient_face, "pred_orient");
            sync_mesh_input->set_input(pred_exp_face, "pred_exp");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<prepare_face_mesh_parameter_node> prepare_mesh_parameter(new prepare_face_mesh_parameter_node());
        prepare_mesh_parameter->set_input(sync_mesh_input->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        face_vertices = inference_mesh->add_output("vertices");
    }
#endif

#if 1
    std::unordered_map<std::string, graph_edge_ptr> hands_vertices;
    for (const auto &part : {"lhand", "rhand"})
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_hands.at(part), "pred_pose");
            sync_mesh_input->set_input(pred_shape_hands.at(part), "pred_shape");
            sync_mesh_input->set_input(pred_orient_hands.at(part), "pred_orient");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<prepare_hand_mesh_parameter_node> prepare_mesh_parameter(new prepare_hand_mesh_parameter_node());
        prepare_mesh_parameter->set_input(sync_mesh_input->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        const auto hand_vertices = inference_mesh->add_output("vertices");
        const auto hand_joints = inference_mesh->add_output("joints");

        hands_vertices.emplace(part, hand_vertices);
    }
#endif

    // iter 1

#if 1
    graph_edge_ptr pred_pose_body1, pred_shape_body1, pred_cam_body1, pred_rotmat_body1;
    {
        std::vector<uint8_t> mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh_aligned_feature_encoder1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_hmr1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features["body"][1], "input");
            sync_mesh_input->set_input(body_vertices, "vertices");
            sync_mesh_input->set_input(pred_cam_body, "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_mesh_aligned_feature(new onnx_runtime_node());
        inference_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_mesh_aligned_feature->set_model_data(mesh_aligned_feature_model_data);
        g->add_node(inference_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_body, "pred_pose.1");
            sync_hmr_input->set_input(pred_shape_body, "pred_shape.1");
            sync_hmr_input->set_input(pred_cam_body, "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hmr_model_data);
        g->add_node(inference_hmr);

        pred_pose_body1 = inference_hmr->add_output("pred_pose");
        pred_shape_body1 = inference_hmr->add_output("pred_shape");
        pred_cam_body1 = inference_hmr->add_output("pred_cam");
        pred_rotmat_body1 = inference_hmr->add_output("pred_rotmat");
    }
#endif

#if 1
    graph_edge_ptr pred_pose_face1, pred_exp_face1, pred_orient_face1, pred_shape_face1, pred_cam_face1;
    {
        std::vector<uint8_t> mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh_aligned_feature_encoder1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_hmr1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_project(new frame_number_sync_node());
        {
            sync_project->set_input(face_joints, "joints");
            sync_project->set_input(pred_cam_body, "cam_sxy");
        }
        g->add_node(sync_project);

        std::shared_ptr<project_joint_iwp_node> project_joints(new project_joint_iwp_node());
        {
            project_joints->set_input(sync_project->get_output());
        }
        g->add_node(project_joints);

        std::shared_ptr<extract_point_2d_node> extract_root(new extract_point_2d_node());
        {
            extract_root->set_index(16);
            extract_root->set_input(project_joints->get_output());
        }
        g->add_node(extract_root);

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features["face"][1], "input");
            sync_mesh_input->set_input(extract_root->get_output(), "center");
            sync_mesh_input->set_input(face_vertices, "vertices");
            sync_mesh_input->set_input(pred_cam_face, "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_mesh_aligned_feature(new onnx_runtime_node());
        inference_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_mesh_aligned_feature->set_model_data(mesh_aligned_feature_model_data);
        g->add_node(inference_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_face, "pred_face.1");
            sync_hmr_input->set_input(pred_exp_face, "pred_exp.1");
            sync_hmr_input->set_input(pred_shape_face, "pred_shape.1");
            sync_hmr_input->set_input(pred_orient_face, "pred_orient.1");
            sync_hmr_input->set_input(pred_cam_face, "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hmr_model_data);
        g->add_node(inference_hmr);

        pred_pose_face1 = inference_hmr->add_output("pred_face");
        pred_exp_face1 = inference_hmr->add_output("pred_exp");
        pred_shape_face1 = inference_hmr->add_output("pred_shape");
        pred_orient_face1 = inference_hmr->add_output("pred_orient");
        pred_cam_face1 = inference_hmr->add_output("pred_cam");
    }
#endif

#if 1
    std::unordered_map<std::string, graph_edge_ptr> pred_pose_hands1, pred_shape_hands1, pred_orient_hands1, pred_cam_hands1;
    for (const auto &part : {"lhand", "rhand"})
    {
        std::vector<uint8_t> mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh_aligned_feature_encoder1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_hmr1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_project(new frame_number_sync_node());
        {
            sync_project->set_input(hands_joints[part], "joints");
            sync_project->set_input(pred_cam_body, "cam_sxy");
        }
        g->add_node(sync_project);

        std::shared_ptr<project_joint_iwp_node> project_joints(new project_joint_iwp_node());
        {
            project_joints->set_input(sync_project->get_output());
        }
        g->add_node(project_joints);

        std::shared_ptr<extract_point_2d_node> extract_root(new extract_point_2d_node());
        {
            extract_root->set_index(0);
            extract_root->set_input(project_joints->get_output());
        }
        g->add_node(extract_root);

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features[part][1], "input");
            sync_mesh_input->set_input(extract_root->get_output(), "center");
            sync_mesh_input->set_input(hands_vertices[part], "vertices");
            sync_mesh_input->set_input(pred_cam_hands[part], "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_mesh_aligned_feature(new onnx_runtime_node());
        inference_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_mesh_aligned_feature->set_model_data(mesh_aligned_feature_model_data);
        g->add_node(inference_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_hands.at(part), "pred_hand.1");
            sync_hmr_input->set_input(pred_shape_hands.at(part), "pred_shape.1");
            sync_hmr_input->set_input(pred_orient_hands.at(part), "pred_orient.1");
            sync_hmr_input->set_input(pred_cam_hands.at(part), "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hmr_model_data);
        g->add_node(inference_hmr);

        const auto pred_pose_hand = inference_hmr->add_output("pred_hand");
        const auto pred_shape_hand = inference_hmr->add_output("pred_shape");
        const auto pred_orient_hand = inference_hmr->add_output("pred_orient");
        const auto pred_cam_hand = inference_hmr->add_output("pred_cam");

        pred_pose_hands1.emplace(part, pred_pose_hand);
        pred_shape_hands1.emplace(part, pred_shape_hand);
        pred_orient_hands1.emplace(part, pred_orient_hand);
        pred_cam_hands1.emplace(part, pred_cam_hand);
    }
#endif

#if 1
    graph_edge_ptr body_vertices1, face_joints1;
    std::unordered_map<std::string, graph_edge_ptr> hands_joints1;
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_body1, "pred_pose");
            sync_mesh_input->set_input(pred_shape_body1, "pred_shape");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_rhand"), "pred_rhand");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_lhand"), "pred_lhand");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_face"), "pred_face");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_exp"), "pred_exp");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<prepare_body_mesh_parameter_node> prepare_mesh_parameter(new prepare_body_mesh_parameter_node());
        prepare_mesh_parameter->set_input(sync_mesh_input->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        body_vertices1 = inference_mesh->add_output("smpl_vertices");
        hands_joints1.emplace("lhand", inference_mesh->add_output("lhand_joints"));
        hands_joints1.emplace("rhand", inference_mesh->add_output("rhand_joints"));
        face_joints1 = inference_mesh->add_output("face_joints");
    }
#endif

#if 1
    graph_edge_ptr face_vertices1;
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_face1, "pred_pose");
            sync_mesh_input->set_input(pred_shape_face1, "pred_shape");
            sync_mesh_input->set_input(pred_orient_face1, "pred_orient");
            sync_mesh_input->set_input(pred_exp_face1, "pred_exp");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<fifo_node> fifo(new fifo_node());
        fifo->set_input(sync_mesh_input->get_output());
        g->add_node(fifo);

        std::shared_ptr<prepare_face_mesh_parameter_node> prepare_mesh_parameter(new prepare_face_mesh_parameter_node());
        prepare_mesh_parameter->set_input(fifo->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        face_vertices1 = inference_mesh->add_output("vertices");
    }
#endif

#if 1
    std::unordered_map<std::string, graph_edge_ptr> hands_vertices1;
    for (const auto &part : {"lhand", "rhand"})
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_hands1.at(part), "pred_pose");
            sync_mesh_input->set_input(pred_shape_hands1.at(part), "pred_shape");
            sync_mesh_input->set_input(pred_orient_hands1.at(part), "pred_orient");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<fifo_node> fifo(new fifo_node());
        fifo->set_input(sync_mesh_input->get_output());
        g->add_node(fifo);

        std::shared_ptr<prepare_hand_mesh_parameter_node> prepare_mesh_parameter(new prepare_hand_mesh_parameter_node());
        prepare_mesh_parameter->set_input(fifo->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        const auto hand_vertices = inference_mesh->add_output("vertices");
        const auto hand_joints = inference_mesh->add_output("joints");

        hands_vertices1.emplace(part, hand_vertices);
    }
#endif

    // iter 2

#if 1
    graph_edge_ptr pred_pose_body2, pred_shape_body2, pred_cam_body2, pred_rotmat_body2;
    {
        std::vector<uint8_t> mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh_aligned_feature_encoder2.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_hmr2.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features["body"][2], "input");
            sync_mesh_input->set_input(body_vertices1, "vertices");
            sync_mesh_input->set_input(pred_cam_body1, "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_mesh_aligned_feature(new onnx_runtime_node());
        inference_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_mesh_aligned_feature->set_model_data(mesh_aligned_feature_model_data);
        g->add_node(inference_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_body1, "pred_pose.1");
            sync_hmr_input->set_input(pred_shape_body1, "pred_shape.1");
            sync_hmr_input->set_input(pred_cam_body1, "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hmr_model_data);
        g->add_node(inference_hmr);

        pred_pose_body2 = inference_hmr->add_output("pred_pose");
        pred_shape_body2 = inference_hmr->add_output("pred_shape");
        pred_cam_body2 = inference_hmr->add_output("pred_cam");
        pred_rotmat_body2 = inference_hmr->add_output("pred_rotmat");
    }
#endif

#if 1
    graph_edge_ptr pred_pose_face2, pred_exp_face2, pred_orient_face2, pred_shape_face2, pred_cam_face2;
    {
        std::vector<uint8_t> mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh_aligned_feature_encoder2.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_hmr2.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_project(new frame_number_sync_node());
        {
            sync_project->set_input(face_joints1, "joints");
            sync_project->set_input(pred_cam_body1, "cam_sxy");
        }
        g->add_node(sync_project);

        std::shared_ptr<project_joint_iwp_node> project_joints(new project_joint_iwp_node());
        {
            project_joints->set_input(sync_project->get_output());
        }
        g->add_node(project_joints);

        std::shared_ptr<extract_point_2d_node> extract_root(new extract_point_2d_node());
        {
            extract_root->set_index(16);
            extract_root->set_input(project_joints->get_output());
        }
        g->add_node(extract_root);

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features["face"][2], "input");
            sync_mesh_input->set_input(extract_root->get_output(), "center");
            sync_mesh_input->set_input(face_vertices1, "vertices");
            sync_mesh_input->set_input(pred_cam_face1, "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_mesh_aligned_feature(new onnx_runtime_node());
        inference_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_mesh_aligned_feature->set_model_data(mesh_aligned_feature_model_data);
        g->add_node(inference_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_face1, "pred_face.1");
            sync_hmr_input->set_input(pred_exp_face1, "pred_exp.1");
            sync_hmr_input->set_input(pred_shape_face1, "pred_shape.1");
            sync_hmr_input->set_input(pred_orient_face1, "pred_orient.1");
            sync_hmr_input->set_input(pred_cam_face1, "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hmr_model_data);
        g->add_node(inference_hmr);

        pred_pose_face2 = inference_hmr->add_output("pred_face");
        pred_exp_face2 = inference_hmr->add_output("pred_exp");
        pred_shape_face2 = inference_hmr->add_output("pred_shape");
        pred_orient_face2 = inference_hmr->add_output("pred_orient");
        pred_cam_face2 = inference_hmr->add_output("pred_cam");
    }
#endif

#if 1
    std::unordered_map<std::string, graph_edge_ptr> pred_pose_hands2, pred_shape_hands2, pred_orient_hands2, pred_cam_hands2;
    for (const auto &part : {"lhand", "rhand"})
    {
        std::vector<uint8_t> mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh_aligned_feature_encoder2.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> hmr_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_hmr2.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hmr_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_project(new frame_number_sync_node());
        {
            sync_project->set_input(hands_joints1[part], "joints");
            sync_project->set_input(pred_cam_body1, "cam_sxy");
        }
        g->add_node(sync_project);

        std::shared_ptr<project_joint_iwp_node> project_joints(new project_joint_iwp_node());
        {
            project_joints->set_input(sync_project->get_output());
        }
        g->add_node(project_joints);

        std::shared_ptr<extract_point_2d_node> extract_root(new extract_point_2d_node());
        {
            extract_root->set_index(0);
            extract_root->set_input(project_joints->get_output());
        }
        g->add_node(extract_root);

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features[part][2], "input");
            sync_mesh_input->set_input(extract_root->get_output(), "center");
            sync_mesh_input->set_input(hands_vertices1[part], "vertices");
            sync_mesh_input->set_input(pred_cam_hands1[part], "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_mesh_aligned_feature(new onnx_runtime_node());
        inference_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_mesh_aligned_feature->set_model_data(mesh_aligned_feature_model_data);
        g->add_node(inference_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_hands1.at(part), "pred_hand.1");
            sync_hmr_input->set_input(pred_shape_hands1.at(part), "pred_shape.1");
            sync_hmr_input->set_input(pred_orient_hands1.at(part), "pred_orient.1");
            sync_hmr_input->set_input(pred_cam_hands1.at(part), "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_hmr(new onnx_runtime_node());
        inference_hmr->set_input(sync_hmr_input->get_output());
        inference_hmr->set_model_data(hmr_model_data);
        g->add_node(inference_hmr);

        const auto pred_pose_hand = inference_hmr->add_output("pred_hand");
        const auto pred_shape_hand = inference_hmr->add_output("pred_shape");
        const auto pred_orient_hand = inference_hmr->add_output("pred_orient");
        const auto pred_cam_hand = inference_hmr->add_output("pred_cam");

        pred_pose_hands2.emplace(part, pred_pose_hand);
        pred_shape_hands2.emplace(part, pred_shape_hand);
        pred_orient_hands2.emplace(part, pred_orient_hand);
        pred_cam_hands2.emplace(part, pred_cam_hand);
    }
#endif

#if 1
    graph_edge_ptr body_vertices2, face_joints2, smplx_vertices;
    std::unordered_map<std::string, graph_edge_ptr> hands_joints2;
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_body2, "pred_pose");
            sync_mesh_input->set_input(pred_shape_body2, "pred_shape");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_rhand"), "pred_rhand");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_lhand"), "pred_lhand");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_face"), "pred_face");
            sync_mesh_input->set_input(initialize_mesh_parameter_obj->add_output("init_exp"), "pred_exp");
            sync_mesh_input->set_input(pred_orient_hands2.at("lhand"), "pred_orient_lhand");
            sync_mesh_input->set_input(pred_orient_hands2.at("rhand"), "pred_orient_rhand");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<optimize_body_mesh_parameter_node> optimize_mesh_parameter(new optimize_body_mesh_parameter_node());
        optimize_mesh_parameter->set_input(sync_mesh_input->get_output());
        optimize_mesh_parameter->set_filename("../sample/pymafx/data/mesh_spec.json");
        g->add_node(optimize_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(optimize_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        body_vertices2 = inference_mesh->add_output("smpl_vertices");
        hands_joints2.emplace("lhand", inference_mesh->add_output("lhand_joints"));
        hands_joints2.emplace("rhand", inference_mesh->add_output("rhand_joints"));
        face_joints2 = inference_mesh->add_output("face_joints");

        smplx_vertices = inference_mesh->add_output("vertices");
    }
#endif

#if 1
    graph_edge_ptr face_vertices2;
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_face2, "pred_pose");
            sync_mesh_input->set_input(pred_shape_face2, "pred_shape");
            sync_mesh_input->set_input(pred_orient_face2, "pred_orient");
            sync_mesh_input->set_input(pred_exp_face2, "pred_exp");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<prepare_face_mesh_parameter_node> prepare_mesh_parameter(new prepare_face_mesh_parameter_node());
        prepare_mesh_parameter->set_input(sync_mesh_input->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        face_vertices2 = inference_mesh->add_output("vertices");
    }
#endif

#if 1
    std::unordered_map<std::string, graph_edge_ptr> hands_vertices2;
    for (const auto &part : {"lhand", "rhand"})
    {
        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(pred_pose_hands2.at(part), "pred_pose");
            sync_mesh_input->set_input(pred_shape_hands2.at(part), "pred_shape");
            sync_mesh_input->set_input(pred_orient_hands2.at(part), "pred_orient");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<prepare_hand_mesh_parameter_node> prepare_mesh_parameter(new prepare_hand_mesh_parameter_node());
        prepare_mesh_parameter->set_input(sync_mesh_input->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh(new onnx_runtime_node());
        inference_mesh->set_input(prepare_mesh_parameter->get_output());
        inference_mesh->set_model_data(mesh_model_data);
        g->add_node(inference_mesh);

        const auto hand_vertices = inference_mesh->add_output("vertices");
        const auto hand_joints = inference_mesh->add_output("joints");

        hands_vertices2.emplace(part, hand_vertices);
    }
#endif

    std::shared_ptr<frame_number_sync_node> sync_output(new frame_number_sync_node());
    {
        sync_output->set_input(pred_cam_body2, "camera");
        sync_output->set_input(smplx_vertices, "vertices");
    }
    g->add_node(sync_output);

    std::shared_ptr<render_node> debug(new render_node());
    debug->set_input(sync_output->get_output());
    g->add_node(debug);

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
