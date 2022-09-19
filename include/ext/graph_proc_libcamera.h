#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <memory>
#include <thread>
#include <atomic>

#include "libcamera_capture.h"
#include "graph_proc_img.h"

namespace coalsack
{
    class libcamera_capture_node : public graph_node
    {
    public:
        enum class option : std::uint32_t
        {
            auto_exposure,
            exposure,
            gain,
        };

    private:
        std::shared_ptr<libcamera_capture> camera;
        graph_edge_ptr output;
        std::vector<std::tuple<option, double>> request_options;
        stream_type stream;
        int width;
        int height;
        int fps;
        image_format format;
        bool emitter_enabled;

        std::mutex mtx;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;
        std::condition_variable cv;
        std::deque<libcamera_capture::buffer> frames;

        static constexpr auto max_frame_queue_size = 10;

    public:
        libcamera_capture_node()
            : graph_node(), output(std::make_shared<graph_edge>(this)), stream(stream_type::COLOR), width(1280), height(720), fps(30), format(image_format::Y8_UINT), emitter_enabled(false)
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "libcamera_capture";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(request_options);
            archive(stream);
            archive(width);
            archive(height);
            archive(fps);
            archive(format);
            archive(emitter_enabled);
        }

        void frame_received(const libcamera_capture::buffer &buffer)
        {
            if (!running)
            {
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mtx);

                if (frames.size() >= max_frame_queue_size)
                {
                    spdlog::error("Fifo overflow");
                }
                else
                {
                    frames.push_back(buffer);
                    cv.notify_one();
                }
            }
        }

        class fps_counter
        {
            uint64_t counter;
            double last_timestamp;
            double fps;

        public:
            fps_counter() : counter(0), last_timestamp(0), fps(0)
            {
            }

            void update(double timestamp)
            {
                if (++counter == 100)
                {
                    fps = counter / (timestamp - last_timestamp) * 1000.0;
                    last_timestamp = timestamp;
                    counter = 0;
                }
            }

            double get_fps() const
            {
                return fps;
            }
        };

        fps_counter counter;

        static libcamera_capture::stream_format get_stream_format(image_format format)
        {
            switch (format)
            {
            case image_format::R8G8B8_UINT:
                return libcamera_capture::stream_format::RGB888;
            case image_format::Y8_UINT:
                return libcamera_capture::stream_format::YUV420;
            case image_format::Y16_UINT:
                return libcamera_capture::stream_format::SBGGR10;
            default:
                throw std::runtime_error("Unsupported format");
            }
        }

        virtual void run() override
        {
            camera.reset(new libcamera_capture());
            camera->open(0);
            camera->configure(libcamera_capture::stream_configuration(get_stream_format(format), width, height, fps));

            for (const auto &[op, value] : request_options)
            {
                switch (op)
                {
                case option::exposure:
                    camera->set_exposure_time(static_cast<std::int32_t>(value));
                    break;
                case option::gain:
                    camera->set_gain(static_cast<float>(value));
                    break;
                case option::auto_exposure:
                    camera->set_auto_exposure(value != 0.0);
                }
            }

            th.reset(new std::thread([this]()
                                     {
            running.store(true);
            while (running.load())
            {
                libcamera_capture::buffer buffer;
                {
                    std::unique_lock<std::mutex> lock(mtx);

                    cv.wait(lock, [&]
                            { return !frames.empty() || !running; });
                    if (!running)
                    {
                        break;
                    }
                    if (!frames.empty())
                    {
                        buffer = frames.front();
                        frames.pop_front();
                    }
                }

                {
                    auto frame = buffer.m;
                    const auto timestamp = buffer.timestamp;
                    const auto frame_number = buffer.frame_number;

                    if (frame.empty())
                    {
                        continue;
                    }

                    counter.update(timestamp);

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
                        stream,
                        0,
                        stream_fmt,
                        camera->get_framerate(),
                        0));
                    msg->set_timestamp(timestamp);
                    msg->set_frame_number(frame_number);

                    output->send(msg);
                }
            } }));

            if (emitter_enabled)
            {
                if (std::system("raspi-gpio set 18 op dh") != 0)
                {
                    spdlog::error("Failed to enable led");
                }
                if (std::system("raspi-gpio set 13 op dh") != 0)
                {
                    spdlog::error("Failed to enable led");
                }
                if (std::system("raspi-gpio set 19 op dh") != 0)
                {
                    spdlog::error("Failed to enable led");
                }
                if (std::system("raspi-gpio set 26 op dh") != 0)
                {
                    spdlog::error("Failed to enable led");
                }
            }

            camera->start(std::bind(&libcamera_capture_node::frame_received, this, std::placeholders::_1));
        }

        virtual void stop() override
        {
            if (!running)
            {
                return;
            }

            running = false;
            cv.notify_all();
            if (th && th->joinable())
            {
                th->join();
            }

            camera->stop();
            camera->close();

            if (emitter_enabled)
            {
                if (std::system("raspi-gpio set 18 op dl") != 0)
                {
                    spdlog::error("Failed to disable led");
                }
                if (std::system("raspi-gpio set 13 op dl") != 0)
                {
                    spdlog::error("Failed to disable led");
                }
                if (std::system("raspi-gpio set 19 op dl") != 0)
                {
                    spdlog::error("Failed to disable led");
                }
                if (std::system("raspi-gpio set 26 op dl") != 0)
                {
                    spdlog::error("Failed to disable led");
                }
            }
        }

        void set_stream(stream_type stream)
        {
            this->stream = stream;
        }

        void set_option(option option, double value)
        {
            request_options.push_back(std::make_tuple(option, value));
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
        void set_format(image_format value)
        {
            format = value;
        }
        void set_emitter_enabled(bool value)
        {
            emitter_enabled = value;
        }
    };
}

CEREAL_REGISTER_TYPE(coalsack::libcamera_capture_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::libcamera_capture_node)
