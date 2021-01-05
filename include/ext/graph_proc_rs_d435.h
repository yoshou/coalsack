#pragma once

#include <vector>
#include <map>
#include <thread>
#include <atomic>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include "graph_proc.h"

using stream_index_pair = std::tuple<rs2_stream, int>;

const stream_index_pair COLOR = {RS2_STREAM_COLOR, 0};
const stream_index_pair DEPTH = {RS2_STREAM_DEPTH, 0};
const stream_index_pair INFRA0 = {RS2_STREAM_INFRARED, 0};
const stream_index_pair INFRA1 = {RS2_STREAM_INFRARED, 1};
const stream_index_pair INFRA2 = {RS2_STREAM_INFRARED, 2};
const stream_index_pair FISHEYE = {RS2_STREAM_FISHEYE, 0};
const stream_index_pair FISHEYE1 = {RS2_STREAM_FISHEYE, 1};
const stream_index_pair FISHEYE2 = {RS2_STREAM_FISHEYE, 2};
const stream_index_pair GYRO = {RS2_STREAM_GYRO, 0};
const stream_index_pair ACCEL = {RS2_STREAM_ACCEL, 0};
const stream_index_pair POSE = {RS2_STREAM_POSE, 0};

const std::vector<stream_index_pair> IMAGE_STREAMS = {DEPTH, INFRA0, INFRA1, INFRA2,
                                                      COLOR,
                                                      FISHEYE,
                                                      FISHEYE1, FISHEYE2};
                                                      
struct stream_profile
{
    stream_profile(rs2_format format = RS2_FORMAT_ANY, rs2_stream stream = RS2_STREAM_ANY,
                    int index = 0, uint32_t width = 0, uint32_t height = 0, uint32_t fps = 0)
        : format(format)
        , stream(stream)
        , index(index)
        , width(width)
        , height(height)
        , fps(fps)
    {}

    rs2_format format;
    rs2_stream stream;
    int index;
    uint32_t width, height, fps;

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(format, stream, index, width, height, fps);
    }
};

static inline bool operator==(const stream_profile &a,
                            const stream_profile &b)
{
    return (a.width == b.width) &&
        (a.height == b.height) &&
        (a.fps == b.fps) &&
        (a.format == b.format) &&
        (a.index == b.index) &&
        (a.stream == b.stream);
}

static inline bool operator<(const stream_profile &lhs,
                            const stream_profile &rhs)
{
    if (lhs.format != rhs.format)
        return lhs.format < rhs.format;
    if (lhs.index != rhs.index)
        return lhs.index < rhs.index;
    return lhs.stream < rhs.stream;
}

class rs_d435_node: public graph_node
{
    graph_edge_ptr output;
    
    std::string serial_number;
    std::vector<std::tuple<stream_index_pair, rs2_option, float>> request_options;
    std::vector<stream_profile> request_profiles;

    // Runtime objects
    rs2::device device;
    std::map<stream_index_pair, rs2::sensor> sensors;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;
public:
    rs_d435_node()
        : graph_node()
        , output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
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
    
    void enable_stream(stream_index_pair stream, int width, int height, rs2_format format = RS2_FORMAT_ANY, int fps = 0)
    {
        rs2_stream stream_type;
        int stream_index;
        std::tie(stream_type, stream_index) = stream;
        stream_profile profile(format, stream_type, stream_index, width, height, fps);
        request_profiles.push_back(profile);
    }

    void enable_stream(stream_index_pair stream)
    {
        enable_stream(stream, 0, 0, RS2_FORMAT_ANY, 0);
    }

    virtual void run() override
    {
        if (request_profiles.size() == 0)
        {
            return;
        }
        
        open();
        
        rs2::config config;
        for (auto profile: request_profiles)
        {
            config.enable_stream(profile.stream, profile.index, profile.width, profile.height, profile.format, profile.fps);
        }
        config.enable_device(get_serial_number());

        th.reset(new std::thread([this, config]() {
            running.store(true);

            rs2::pipeline pipeline;
            auto pipeline_profile = pipeline.start(config);
            while (running.load())
            {
                rs2::frameset frameset = pipeline.wait_for_frames();
                frame_callback(frameset);
            }
            pipeline.stop();
        }));
    }

    virtual void stop() override
    {
        if (running.load())
        {
            running.store(false);
            th->join();
        }
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(serial_number);
        archive(request_options);
        archive(request_profiles);
    }
    
private:

    void video_frame_callback(rs2::video_frame frame)
    {
        auto msg = std::make_shared<frame_message<image>>();

        image img(frame.get_width(), frame.get_height(),frame.get_bytes_per_pixel() * 8,
            frame.get_stride_in_bytes(), (const uint8_t*)frame.get_data());

        msg->set_data(std::move(img));
        output->send(msg);
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
        for (auto option_req: request_options)
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
                stream_index_pair sip(profile.stream_type(), profile.stream_index());
                if (sensors.find(sip) == sensors.end())
                {
                    sensors[sip] = sensor;
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(rs_d435_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, rs_d435_node)
