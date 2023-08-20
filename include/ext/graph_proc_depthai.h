#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <mutex>
#include <memory>
#include <thread>
#include <atomic>

#include "graph_proc_img.h"

#ifdef ENABLE_DEPTHAI_EXT
#include <depthai/depthai.hpp>
#endif

namespace coalsack
{
    class depthai_color_camera_node : public graph_node
    {
    private:
        graph_edge_ptr output;
        int width;
        int height;
        int fps;

        std::mutex mtx;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;

        static constexpr auto max_frame_queue_size = 10;

    public:
        depthai_color_camera_node()
            : graph_node(), output(std::make_shared<graph_edge>(this)), width(1280), height(720), fps(30)
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "depthai_color_camera";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(width);
            archive(height);
            archive(fps);
        }

        void set_width(int value)
        {
            width = value;
        }
        void set_height(int value)
        {
            height = value;
        }
        void set_fps(int value)
        {
            fps = value;
        }

        virtual void run() override
        {
            th.reset(new std::thread([this]()
                                     {
            running.store(true);

#ifdef ENABLE_DEPTHAI_EXT
            dai::Pipeline pipeline;

            auto camera = pipeline.create<dai::node::ColorCamera>();
            auto xout_video = pipeline.create<dai::node::XLinkOut>();

            {
                xout_video->setStreamName("video");
                xout_video->input.setBlocking(false);
                xout_video->input.setQueueSize(max_frame_queue_size);
            }

            {
                camera->setBoardSocket(dai::CameraBoardSocket::CAM_A);
                camera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
                camera->setVideoSize(width, height);
                camera->setFps(fps);
                camera->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
            }

            camera->video.link(xout_video->input);

            dai::Device device(pipeline);

            auto video = device.getOutputQueue("video");

            const auto system_clock_start = std::chrono::system_clock::now();
            const auto steady_clock_start = std::chrono::steady_clock::now();

            while (running.load())
            {
                auto video_in = video->get<dai::ImgFrame>();

                auto frame = video_in->getCvFrame();
                const auto frame_number = video_in->getSequenceNum();

                const auto timestamp = (std::chrono::duration_cast<std::chrono::nanoseconds>(system_clock_start.time_since_epoch()).count() +
                                        (std::chrono::duration_cast<std::chrono::nanoseconds>(video_in->getTimestamp().time_since_epoch()).count() -
                                            std::chrono::duration_cast<std::chrono::nanoseconds>(steady_clock_start.time_since_epoch()).count())) /
                                        1000000.0;

                if (frame.empty())
                {
                    continue;
                }

                auto msg = std::make_shared<frame_message<image>>();

                image img(static_cast<std::uint32_t>(frame.size().width),
                            static_cast<std::uint32_t>(frame.size().height),
                            static_cast<std::uint32_t>(frame.elemSize()),
                            static_cast<std::uint32_t>(frame.step), (const uint8_t *)frame.data);

                if (frame.channels() == 1)
                {
                    if (frame.elemSize() == 1)
                    {
                        img.set_format(image_format::Y8_UINT);
                    }
                    else if (frame.elemSize() == 2)
                    {
                        img.set_format(image_format::Y16_UINT);
                    }
                }
                else if (frame.channels() == 3)
                {
                    img.set_format(image_format::B8G8R8_UINT);
                }
                else if (frame.channels() == 4)
                {
                    img.set_format(image_format::B8G8R8A8_UINT);
                }
                stream_format stream_fmt = stream_format::ANY;
                if (frame.channels() == 1)
                {
                    if (frame.elemSize() == 1)
                    {
                        stream_fmt = stream_format::Y8;
                    }
                    else if (frame.elemSize() == 2)
                    {
                        stream_fmt = stream_format::Y16;
                    }
                }
                else if (frame.channels() == 3)
                {
                    stream_fmt = stream_format::BGR8;
                }
                else if (frame.channels() == 4)
                {
                    stream_fmt = stream_format::BGRA8;
                }

                msg->set_data(std::move(img));
                msg->set_profile(std::make_shared<stream_profile>(
                    stream_type::COLOR,
                    0,
                    stream_fmt,
                    camera->getFps(),
                    0));
                msg->set_timestamp(timestamp);
                msg->set_frame_number(frame_number);

                output->send(msg);
            }
#endif
            }));
        }

        virtual void stop() override
        {
            if (!running)
            {
                return;
            }

            running = false;
            if (th && th->joinable())
            {
                th->join();
            }
        }
    };
}

CEREAL_REGISTER_TYPE(coalsack::depthai_color_camera_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::depthai_color_camera_node)
