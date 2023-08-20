#pragma once

#include <vector>
#include <map>
#include <thread>
#include <atomic>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include "graph_proc.h"

namespace coalsack
{
    enum class rs2_stream_type : std::uint32_t
    {
        ANY,
        DEPTH,
        COLOR,
        INFRARED,
        FISHEYE,
        GYRO,
        ACCEL,
        GPIO,
        POSE,
        CONFIDENCE,
    };

    enum class rs2_format_type : std::uint32_t
    {
        ANY,
        Z16,
        DISPARITY16,
        XYZ32F,
        YUYV,
        RGB8,
        BGR8,
        RGBA8,
        BGRA8,
        Y8,
        Y16,
        RAW10,
        RAW16,
        RAW8,
        UYVY,
        MOTION_RAW,
        MOTION_XYZ32F,
        GPIO_RAW,
        SIX_DOF,
        DISPARITY32,
        Y10BPACK,
        DISTANCE,
        MJPEG,
        Y8I,
        Y12I,
        INZI,
        INVI,
        W10,
        Z16H,
        FG,
        Y411,
    };

    using stream_index_pair = std::tuple<rs2_stream_type, int>;

    const stream_index_pair COLOR = {rs2_stream_type::COLOR, 0};
    const stream_index_pair DEPTH = {rs2_stream_type::DEPTH, 0};
    const stream_index_pair INFRA0 = {rs2_stream_type::INFRARED, 0};
    const stream_index_pair INFRA1 = {rs2_stream_type::INFRARED, 1};
    const stream_index_pair INFRA2 = {rs2_stream_type::INFRARED, 2};
    const stream_index_pair FISHEYE = {rs2_stream_type::FISHEYE, 0};
    const stream_index_pair FISHEYE1 = {rs2_stream_type::FISHEYE, 1};
    const stream_index_pair FISHEYE2 = {rs2_stream_type::FISHEYE, 2};
    const stream_index_pair GYRO = {rs2_stream_type::GYRO, 0};
    const stream_index_pair ACCEL = {rs2_stream_type::ACCEL, 0};
    const stream_index_pair POSE = {rs2_stream_type::POSE, 0};

    const std::vector<stream_index_pair> IMAGE_STREAMS = {DEPTH, INFRA0, INFRA1, INFRA2,
                                                          COLOR,
                                                          FISHEYE,
                                                          FISHEYE1, FISHEYE2};

    struct stream_profile_request
    {
        stream_profile_request(rs2_format_type format = rs2_format_type::ANY, rs2_stream_type stream = rs2_stream_type::ANY,
                               int index = 0, uint32_t width = 0, uint32_t height = 0, uint32_t fps = 0)
            : format(format), stream(stream), index(index), width(width), height(height), fps(fps)
        {
        }

        rs2_format_type format;
        rs2_stream_type stream;
        int index;
        uint32_t width, height, fps;

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(format, stream, index, width, height, fps);
        }
    };

    static inline bool operator==(const stream_profile_request &a,
                                  const stream_profile_request &b)
    {
        return (a.width == b.width) &&
               (a.height == b.height) &&
               (a.fps == b.fps) &&
               (a.format == b.format) &&
               (a.index == b.index) &&
               (a.stream == b.stream);
    }

    static inline bool operator<(const stream_profile_request &lhs,
                                 const stream_profile_request &rhs)
    {
        if (lhs.format != rhs.format)
            return lhs.format < rhs.format;
        if (lhs.index != rhs.index)
            return lhs.index < rhs.index;
        return lhs.stream < rhs.stream;
    }

    static stream_format convert_stream_format(rs2_format format)
    {
        switch (format)
        {
        case RS2_FORMAT_Z16:
            return stream_format::Z16;
        case RS2_FORMAT_RGB8:
            return stream_format::RGB8;
        case RS2_FORMAT_BGR8:
            return stream_format::BGR8;
        case RS2_FORMAT_RGBA8:
            return stream_format::RGBA8;
        case RS2_FORMAT_BGRA8:
            return stream_format::BGRA8;
        case RS2_FORMAT_Y8:
            return stream_format::Y8;
        case RS2_FORMAT_Y16:
            return stream_format::Y16;
        case RS2_FORMAT_YUYV:
            return stream_format::YUYV;
        case RS2_FORMAT_UYVY:
            return stream_format::UYVY;
        default:
            return stream_format::ANY;
        }
    }

    static image_format convert_image_format(rs2_format format)
    {
        switch (format)
        {
        case RS2_FORMAT_Z16:
            return image_format::Z16_UINT;
        case RS2_FORMAT_RGB8:
            return image_format::R8G8B8_UINT;
        case RS2_FORMAT_BGR8:
            return image_format::B8G8R8_UINT;
        case RS2_FORMAT_RGBA8:
            return image_format::R8G8B8A8_UINT;
        case RS2_FORMAT_BGRA8:
            return image_format::B8G8R8A8_UINT;
        case RS2_FORMAT_Y8:
            return image_format::Y8_UINT;
        case RS2_FORMAT_Y16:
            return image_format::Y16_UINT;
        case RS2_FORMAT_YUYV:
            return image_format::YUY2;
        case RS2_FORMAT_UYVY:
            return image_format::UYVY;
        default:
            return image_format::ANY;
        }
    }

    static stream_type convert_stream_type(rs2_stream type)
    {
        switch (type)
        {
        case RS2_STREAM_DEPTH:
            return stream_type::DEPTH;
        case RS2_STREAM_COLOR:
            return stream_type::COLOR;
        case RS2_STREAM_INFRARED:
            return stream_type::INFRARED;
        default:
            return stream_type::ANY;
        }
    }

    class rs_d435_node : public graph_node
    {
        std::string serial_number;
        std::vector<std::tuple<stream_index_pair, rs2_option, float>> request_options;
        std::vector<stream_profile_request> request_profiles;

        // Runtime objects
        rs2::device device;
        std::map<stream_index_pair, rs2::sensor> sensors;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;

        static int64_t get_stream_profile_request_id(stream_profile_request profile)
        {
            assert(profile.index >= 0);

            const auto stream = convert_to_rs2_stream(profile.stream);
            const auto format = convert_to_rs2_format(profile.format);

            return stream * pow(10, 12) + format * pow(10, 10) + profile.fps * pow(10, 8) +
                   profile.width * pow(10, 4) + profile.height + (uint32_t)profile.index;
        }

        static int64_t get_stream_profile_id(rs2::stream_profile profile)
        {
            assert(profile.stream_index() >= 0);

            int64_t key = profile.stream_type() * pow(10, 12) + profile.format() * pow(10, 10) + profile.fps() * pow(10, 8) + profile.stream_index();
            if (profile.is<rs2::video_stream_profile>())
            {
                rs2::video_stream_profile video_profile = profile.as<rs2::video_stream_profile>();
                key += video_profile.width() * pow(10, 4) + video_profile.height();
            }
            return key;
        }

        static rs2_stream convert_to_rs2_stream(rs2_stream_type type)
        {
            switch (type)
            {
            case rs2_stream_type::DEPTH:
                return RS2_STREAM_DEPTH;
            case rs2_stream_type::COLOR:
                return RS2_STREAM_COLOR;
            case rs2_stream_type::INFRARED:
                return RS2_STREAM_INFRARED;
            default:
                throw std::runtime_error("Unsupported");
            }
        }

        static rs2_stream_type convert_from_rs2_stream(rs2_stream type)
        {
            switch (type)
            {
            case RS2_STREAM_DEPTH:
                return rs2_stream_type::DEPTH;
            case  RS2_STREAM_COLOR:
                return rs2_stream_type::COLOR;
            case  RS2_STREAM_INFRARED:
                return rs2_stream_type::INFRARED;
            default:
                throw std::runtime_error("Unsupported");
            }
        }

        static rs2_format convert_to_rs2_format(rs2_format_type type)
        {
            switch (type)
            {
            case rs2_format_type::Z16:
                return RS2_FORMAT_Z16;
            case rs2_format_type::RGB8:
                return RS2_FORMAT_RGB8;
            case rs2_format_type::BGR8:
                return RS2_FORMAT_BGR8;
            case rs2_format_type::RGBA8:
                return RS2_FORMAT_RGBA8;
            case rs2_format_type::BGRA8:
                return RS2_FORMAT_BGRA8;
            case rs2_format_type::Y8:
                return RS2_FORMAT_Y8;
            case rs2_format_type::Y16:
                return RS2_FORMAT_Y16;
            case rs2_format_type::YUYV:
                return RS2_FORMAT_YUYV;
            case rs2_format_type::UYVY:
                return RS2_FORMAT_UYVY;
            default:
                throw std::runtime_error("Unsupported");
            }
        }

        static rs2_format_type convert_from_rs2_format(rs2_format type)
        {
            switch (type)
            {
            case RS2_FORMAT_Z16:
                return rs2_format_type::Z16;
            case RS2_FORMAT_RGB8:
                return rs2_format_type::RGB8;
            case RS2_FORMAT_BGR8:
                return rs2_format_type::BGR8;
            case RS2_FORMAT_RGBA8:
                return rs2_format_type::RGBA8;
            case RS2_FORMAT_BGRA8:
                return rs2_format_type::BGRA8;
            case RS2_FORMAT_Y8:
                return rs2_format_type::Y8;
            case RS2_FORMAT_Y16:
                return rs2_format_type::Y16;
            case RS2_FORMAT_YUYV:
                return rs2_format_type::YUYV;
            case RS2_FORMAT_UYVY:
                return rs2_format_type::UYVY;
            default:
                throw std::runtime_error("Unsupported");
            }
        }

    public:
        rs_d435_node()
            : graph_node()
        {
        }

        virtual std::string get_proc_name() const override
        {
            return "rs_d435";
        }

        void set_serial_number(std::string serial_number)
        {
            this->serial_number = serial_number;
        }

        std::string get_serial_number() const
        {
            return serial_number;
        }

        graph_edge_ptr get_output(stream_index_pair stream, int width, int height, rs2_format_type format = rs2_format_type::ANY, int fps = 0)
        {
            const auto [stream_type, stream_index] = stream;
            stream_profile_request profile(format, stream_type, stream_index, width, height, fps);
            std::string output_name = "stream" + std::to_string(get_stream_profile_request_id(profile));

            return graph_node::get_output(output_name);
        }

        graph_edge_ptr add_output(stream_index_pair stream, int width = 0, int height = 0, rs2_format_type format = rs2_format_type::ANY, int fps = 0)
        {
            const auto [stream_type, stream_index] = stream;
            stream_profile_request profile(format, stream_type, stream_index, width, height, fps);
            std::string output_name = "stream" + std::to_string(get_stream_profile_request_id(profile));

            auto outputs = get_outputs();
            auto it = outputs.find(output_name);
            if (it == outputs.end())
            {
                request_profiles.push_back(profile);

                auto output = std::make_shared<graph_edge>(this);
                set_output(output, output_name);
                return output;
            }
            return it->second;
        }

        void set_option(stream_index_pair stream, rs2_option option, float value)
        {
            request_options.push_back(std::make_tuple(stream, option, value));
        }

        virtual void run() override
        {
            if (request_profiles.size() == 0)
            {
                return;
            }

            open();
            configure();

            rs2::config config;
            for (auto profile : request_profiles)
            {
                const auto stream = convert_to_rs2_stream(profile.stream);
                const auto format = convert_to_rs2_format(profile.format);
                config.enable_stream(stream, profile.index, profile.width, profile.height, format, profile.fps);
            }
            config.enable_device(get_serial_number());

            th.reset(new std::thread([this, config]()
                                     {
            running.store(true);

            rs2::pipeline pipeline;
            auto pipeline_profile = pipeline.start(config);
            while (running.load())
            {
                rs2::frameset frameset = pipeline.wait_for_frames();
                frame_callback(frameset);
            }
            pipeline.stop(); }));
        }

        virtual void stop() override
        {
            if (running.load())
            {
                running.store(false);
                th->join();
            }
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
            archive(serial_number);
            archive(request_options);
            archive(request_profiles);
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

            archive(serial_number);
            archive(request_options);
            archive(request_profiles);
        }

    private:
        void video_frame_callback(rs2::video_frame frame)
        {
            auto profile = frame.get_profile();

            auto msg = std::make_shared<frame_message<image>>();

            image img(frame.get_width(), frame.get_height(), frame.get_bytes_per_pixel(),
                      frame.get_stride_in_bytes(), (const uint8_t *)frame.get_data());

            img.set_format(convert_image_format(profile.format()));

            msg->set_data(std::move(img));
            msg->set_profile(std::make_shared<stream_profile>(
                convert_stream_type(profile.stream_type()),
                profile.stream_index(),
                convert_stream_format(profile.format()),
                profile.fps(),
                profile.unique_id()));
            msg->set_timestamp(frame.get_timestamp());
            msg->set_frame_number(frame.get_frame_number());

            if (auto video_profile = profile.as<rs2::video_stream_profile>())
            {
                const auto stream = convert_from_rs2_stream(profile.stream_type());
                const auto format = convert_from_rs2_format(profile.format());
                
                stream_index_pair sip = {stream, profile.stream_index()};
                auto output = get_output(sip, video_profile.width(), video_profile.height(), format, profile.fps());
                output->send(msg);
            }
        }

        void frame_callback(rs2::frame frame)
        {
            if (frame.is<rs2::frameset>())
            {
                auto frameset = frame.as<rs2::frameset>();
                for (auto it = frameset.begin(); it != frameset.end(); ++it)
                {
                    auto f = (*it);
                    frame_callback(f);
                }
            }
            else if (frame.is<rs2::video_frame>())
            {
                video_frame_callback(frame);
            }
        }

        void configure()
        {
            for (auto option_req : request_options)
            {
                stream_index_pair sip;
                rs2_option option;
                float value;
                std::tie(sip, option, value) = option_req;

                auto sensor = sensors[sip];
                sensor.set_option(option, value);
            }
        }

        void open()
        {
            rs2::context ctx;
            auto devices = ctx.query_devices();

            if (devices.size() == 0)
            {
                throw std::runtime_error("Not found D435 device");
            }

            if (serial_number.empty())
            {
                device = devices.front();
                serial_number = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            }
            else
            {
                for (auto &&dev : devices)
                {
                    if (serial_number == dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))
                    {
                        device = dev;
                        break;
                    }
                }
            }

            if (!device)
            {
                throw std::runtime_error("Not found D435 device");
            }

            for (auto &&sensor : device.query_sensors())
            {
                for (auto &profile : sensor.get_stream_profiles())
                {
                    const auto stream = convert_from_rs2_stream(profile.stream_type());
                    stream_index_pair sip(stream, profile.stream_index());
                    if (sensors.find(sip) == sensors.end())
                    {
                        sensors[sip] = sensor;
                    }
                }
            }
        }
    };
}

CEREAL_REGISTER_TYPE(coalsack::rs_d435_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::rs_d435_node)
