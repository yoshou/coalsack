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

class onnx_runtime_node : public graph_node
{
    std::vector<uint8_t> model_data;

    Ort::Session session;
    std::vector<std::string> input_names;
    std::unordered_map<std::string, std::vector<int64_t>> input_dims;
    std::unordered_map<std::string, int> input_types;

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

    Ort::Env env { ORT_LOGGING_LEVEL_WARNING };

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
        options.gpu_mem_limit = 8ULL * 1024 * 1024 * 1024;
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
            input_names.push_back(input_name.get());

            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            const auto type = tensor_info.GetElementType();
            input_types[input_name.get()] = type;

            const auto input_shape = tensor_info.GetShape();
            input_dims[input_name.get()] = input_shape;
        }
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

        std::vector<const char *> output_name_strs;
        std::vector<Ort::Value> output_tensors;
        std::shared_ptr<frame_message_base> base_frame_msg = nullptr;

        {
            const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<const char *> input_name_strs;
            std::vector<Ort::Value> input_tensors;

            std::lock_guard<std::mutex> lock(mtx);
            if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(message))
            {
                assert(input_names.size() == 1);
                {
                    const auto name = input_names.at(0);
                    const auto dims = input_dims.at(name);
                    const auto type = input_types.at(name);
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
                for (const auto &name : input_names)
                {
                    const auto dims = input_dims.at(name);
                    const auto type = input_types.at(name);
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

            assert(input_name_strs.size() == input_names.size());
            assert(input_tensors.size() == input_names.size());

            for (const auto &[name, _] : get_outputs())
            {
                output_name_strs.push_back(name.c_str());
            }

            output_tensors = session.Run(Ort::RunOptions{nullptr}, input_name_strs.data(), input_tensors.data(), input_tensors.size(), output_name_strs.data(), output_name_strs.size());

            // Free memory explicitly
            input_tensors.clear();
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
            const auto x0 = src.at({0, j, i});
            const auto x1 = src.at({2, j, i});
            const auto x2 = src.at({4, j, i});
            const auto x3 = src.at({1, j, i});
            const auto x4 = src.at({3, j, i});
            const auto x5 = src.at({5, j, i});

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
            dst.set({1, 2, j, i}, src.get({1, 1, j, i}));
            dst.set({2, 2, j, i}, src.get({1, 1, j, i}));
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
        data_loader->set_interval(10000);
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

        std::vector<uint8_t> body_hmr0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_hmr0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            body_hmr0_model_data = std::move(data);
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

        std::shared_ptr<onnx_runtime_node> inference_body_hmr0(new onnx_runtime_node());
        inference_body_hmr0->set_input(sync1->get_output());
        inference_body_hmr0->set_model_data(body_hmr0_model_data);
        g->add_node(inference_body_hmr0);

        pred_pose_body = inference_body_hmr0->add_output("pred_pose");
        pred_shape_body = inference_body_hmr0->add_output("pred_shape");
        pred_cam_body = inference_body_hmr0->add_output("pred_cam");
        pred_rotmat_body = inference_body_hmr0->add_output("pred_rotmat");
    }
#endif

#if 1
    graph_edge_ptr pred_pose_face, pred_exp_face, pred_orient_face, pred_shape_face, pred_cam_face;
    {
#if 0
        std::vector<uint8_t> face_mesh0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            face_mesh0_model_data = std::move(data);
        }
#endif

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
    std::unordered_map<std::string, graph_edge_ptr> pred_pose_hands, pred_shape_hands, pred_orient_hands;
    for (const auto& part : {"lhand", "rhand"})
    {
#if 0
        std::vector<uint8_t> hand_mesh0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            hand_mesh0_model_data = std::move(data);
        }
#endif

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

        std::shared_ptr<onnx_runtime_node> inference_hand_hmr0(new onnx_runtime_node());
        inference_hand_hmr0->set_input(sync_hmr_input->get_output());
        inference_hand_hmr0->set_model_data(hand_hmr0_model_data);
        g->add_node(inference_hand_hmr0);

        const auto pred_pose_hand = inference_hand_hmr0->add_output("pred_hand");
        const auto pred_shape_hand = inference_hand_hmr0->add_output("pred_shape");
        const auto pred_orient_hand = inference_hand_hmr0->add_output("pred_orient");
        inference_hand_hmr0->add_output("pred_cam");

        pred_pose_hands.emplace(part, pred_pose_hand);
        pred_shape_hands.emplace(part, pred_shape_hand);
        pred_orient_hands.emplace(part, pred_orient_hand);
    }
#endif

#if 1
    graph_edge_ptr body_vertices;
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

        std::shared_ptr<fifo_node> fifo(new fifo_node());
        fifo->set_input(sync_mesh_input->get_output());
        g->add_node(fifo);

        std::shared_ptr<prepare_body_mesh_parameter_node> prepare_mesh_parameter(new prepare_body_mesh_parameter_node());
        prepare_mesh_parameter->set_input(fifo->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh0_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh0(new onnx_runtime_node());
        inference_mesh0->set_input(prepare_mesh_parameter->get_output());
        inference_mesh0->set_model_data(mesh0_model_data);
        g->add_node(inference_mesh0);

        body_vertices = inference_mesh0->add_output("smpl_vertices");
    }
#endif

#if 1
    graph_edge_ptr pred_pose_body1, pred_shape_body1, pred_cam_body1, pred_rotmat_body1;
    {
        std::vector<uint8_t> body_mesh_aligned_feature_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_mesh_aligned_feature_encoder1.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            body_mesh_aligned_feature_model_data = std::move(data);
        }

        std::vector<uint8_t> body_hmr0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/body_hmr0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            body_hmr0_model_data = std::move(data);
        }

        std::shared_ptr<frame_number_sync_node> sync_mesh_input(new frame_number_sync_node());
        {
            sync_mesh_input->set_input(part_features["body"][1], "input");
            sync_mesh_input->set_input(body_vertices, "vertices");
            sync_mesh_input->set_input(pred_cam_body, "pred_cam");
        }
        g->add_node(sync_mesh_input);

        std::shared_ptr<onnx_runtime_node> inference_body_mesh_aligned_feature(new onnx_runtime_node());
        inference_body_mesh_aligned_feature->set_input(sync_mesh_input->get_output());
        inference_body_mesh_aligned_feature->set_model_data(body_mesh_aligned_feature_model_data);
        g->add_node(inference_body_mesh_aligned_feature);

        std::shared_ptr<frame_number_sync_node> sync_hmr_input(new frame_number_sync_node());
        {
            sync_hmr_input->set_input(inference_body_mesh_aligned_feature->add_output("output"), "x");
            sync_hmr_input->set_input(pred_pose_body, "pred_pose.1");
            sync_hmr_input->set_input(pred_shape_body, "pred_shape.1");
            sync_hmr_input->set_input(pred_cam_body, "pred_cam.1");
        }
        g->add_node(sync_hmr_input);

        std::shared_ptr<onnx_runtime_node> inference_body_hmr(new onnx_runtime_node());
        inference_body_hmr->set_input(sync_hmr_input->get_output());
        inference_body_hmr->set_model_data(body_hmr0_model_data);
        g->add_node(inference_body_hmr);

        pred_pose_body1 = inference_body_hmr->add_output("pred_pose");
        pred_shape_body1 = inference_body_hmr->add_output("pred_shape");
        pred_cam_body1 = inference_body_hmr->add_output("pred_cam");
        pred_rotmat_body1 = inference_body_hmr->add_output("pred_rotmat");
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

        std::shared_ptr<fifo_node> fifo(new fifo_node());
        fifo->set_input(sync_mesh_input->get_output());
        g->add_node(fifo);

        std::shared_ptr<prepare_face_mesh_parameter_node> prepare_mesh_parameter(new prepare_face_mesh_parameter_node());
        prepare_mesh_parameter->set_input(fifo->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/face_mesh0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh0_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh0(new onnx_runtime_node());
        inference_mesh0->set_input(prepare_mesh_parameter->get_output());
        inference_mesh0->set_model_data(mesh0_model_data);
        g->add_node(inference_mesh0);

        face_vertices = inference_mesh0->add_output("vertices");
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

        std::shared_ptr<fifo_node> fifo(new fifo_node());
        fifo->set_input(sync_mesh_input->get_output());
        g->add_node(fifo);

        std::shared_ptr<prepare_hand_mesh_parameter_node> prepare_mesh_parameter(new prepare_hand_mesh_parameter_node());
        prepare_mesh_parameter->set_input(fifo->get_output());
        g->add_node(prepare_mesh_parameter);

        std::vector<uint8_t> mesh0_model_data;
        {
            const auto model_path = "../sample/pymafx/data/hand_mesh0.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            mesh0_model_data = std::move(data);
        }

        std::shared_ptr<onnx_runtime_node> inference_mesh0(new onnx_runtime_node());
        inference_mesh0->set_input(prepare_mesh_parameter->get_output());
        inference_mesh0->set_model_data(mesh0_model_data);
        g->add_node(inference_mesh0);

        const auto hand_vertices = inference_mesh0->add_output("vertices");

        hands_vertices.emplace(part, hand_vertices);
    }
#endif

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
