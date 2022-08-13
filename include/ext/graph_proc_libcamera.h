#pragma once

#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <memory>
#include <thread>
#include <atomic>
#include <iostream>
#include <iomanip>

#ifdef ENABLE_LIBCAMERA_EXT

#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/property_ids.h>

#include <sys/mman.h>

#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "graph_proc_img.h"

namespace coalsack
{
    class libcamera_capture
    {
    public:
        struct buffer
        {
            cv::Mat m;
            double timestamp;
            std::uint32_t frame_number;
        };

        struct stream_configuration
        {
            stream_configuration(image_format format, uint32_t width = 0, uint32_t height = 0, uint32_t fps = 0)
                : format(format), width(width), height(height), fps(fps)
            {
            }

            image_format format;
            uint32_t width, height, fps;

            template <typename Archive>
            void serialize(Archive &archive)
            {
                archive(format, width, height, fps);
            }
        };

        struct mapped_buffer_data
        {
            void *pointer;
            size_t length;
        };

    private:
#ifdef ENABLE_LIBCAMERA_EXT
        std::shared_ptr<libcamera::Camera> camera;
        std::unique_ptr<libcamera::CameraManager> camera_manager;
        std::unique_ptr<libcamera::FrameBufferAllocator> allocator;

        std::queue<libcamera::Request *> res;
        std::mutex mtx;
        std::condition_variable cv;
        std::shared_ptr<std::thread> th;
        std::atomic_bool running;

        std::unique_ptr<libcamera::CameraConfiguration> config;
        libcamera::Stream *stream;
        libcamera::ControlList controls;
        std::vector<std::unique_ptr<libcamera::Request>> requests;
        std::unordered_map<int, mapped_buffer_data> mapped_buffer;
        int fps;

        std::function<void(const libcamera_capture::buffer &)> frame_received;

        void request_completed(libcamera::Request *request)
        {
            if (request->status() == libcamera::Request::RequestCancelled)
            {
                return;
            }

            // std::lock_guard<std::mutex> lock(mtx);
            // res.push(request);
            process_res(request);
            // cv.notify_one();
        }

        void process_res(libcamera::Request *request)
        {
            const libcamera::Request::BufferMap &buffers = request->buffers();
            for (auto buffer_pair : buffers)
            {
                const libcamera::Stream *stream = buffer_pair.first;
                const auto &cfg = stream->configuration();
                libcamera::FrameBuffer *buffer = buffer_pair.second;
                const libcamera::FrameMetadata &metadata = buffer->metadata();

                // const double timestamp = static_cast<double>(metadata.timestamp) / 1000000.0;
                const double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                const std::uint32_t sequence = metadata.sequence;

                cv::Mat frame;
                if (cfg.pixelFormat == libcamera::formats::YUV420)
                {
                    std::vector<cv::Mat> planes;

                    void *y_ptr = mapped_buffer.at(buffer->planes().at(0).fd.get()).pointer;
                    void *u_ptr = mapped_buffer.at(buffer->planes().at(1).fd.get()).pointer;
                    void *v_ptr = mapped_buffer.at(buffer->planes().at(2).fd.get()).pointer;

                    frame = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC1, y_ptr, cfg.stride);
                }
                else if (cfg.pixelFormat == libcamera::formats::RGB888)
                {
                    std::vector<cv::Mat> planes;

                    void *ptr = mapped_buffer.at(buffer->planes().at(0).fd.get()).pointer;
                    frame = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC3, ptr, cfg.stride);
                }

                this->frame_received(libcamera_capture::buffer{frame, timestamp, sequence});
            }

            /* Re-queue the Request to the camera. */
            request->reuse(libcamera::Request::ReuseBuffers);
            camera->queueRequest(request);
        }

        static libcamera::PixelFormat conv_format(image_format format)
        {
            switch (format)
            {
            case image_format::R8G8B8_UINT:
                return libcamera::formats::RGB888;
            case image_format::Y8_UINT:
                return libcamera::formats::YUV420;
            default:
                throw std::runtime_error("Unsupported format");
            }
        }

        void allocate_buffers()
        {
            for (libcamera::StreamConfiguration &cfg : *config)
            {
                libcamera::Stream *stream = cfg.stream();

                if (allocator->allocate(stream) < 0)
                {
                    throw std::runtime_error("Failed to allocate capture buffers");
                }
            }
        }

        void set_framerate(int32_t fps)
        {
            this->fps = fps;

            const int64_t frame_time = 1000000 / fps; // in us
            controls.set(libcamera::controls::FrameDurationLimits, {frame_time, frame_time});
        }

        void set_default_control()
        {
            const float roi_width = 0;
            const float roi_height = 0;
            const float roi_x = 0;
            const float roi_y = 0;

            if (!controls.contains(libcamera::controls::ScalerCrop) && roi_width != 0 && roi_height != 0)
            {
                libcamera::Rectangle sensor_area = camera->properties().get(libcamera::properties::ScalerCropMaximum);
                int x = roi_x * sensor_area.width;
                int y = roi_y * sensor_area.height;
                int w = roi_width * sensor_area.width;
                int h = roi_height * sensor_area.height;
                libcamera::Rectangle crop(x, y, w, h);
                crop.translateBy(sensor_area.topLeft());
                controls.set(libcamera::controls::ScalerCrop, crop);
            }
        }

        void make_requests()
        {
            const std::vector<std::unique_ptr<libcamera::FrameBuffer>> &buffers = allocator->buffers(stream);
            for (unsigned int i = 0; i < buffers.size(); ++i)
            {
                std::unique_ptr<libcamera::Request> request = camera->createRequest();
                if (!request)
                {
                    throw std::runtime_error("Failed to make request");
                }

                const std::unique_ptr<libcamera::FrameBuffer> &buffer = buffers[i];
                if (request->addBuffer(stream, buffer.get()) < 0)
                {
                    throw std::runtime_error("Failed to add buffer to request");
                }

                request->controls() = controls;
                requests.push_back(std::move(request));

                for (auto plane : buffer->planes())
                {
                    void *ptr = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                    if (!ptr)
                    {
                        throw std::runtime_error("Failed to map the buffer");
                    }
                    mapped_buffer.emplace(plane.fd.get(), mapped_buffer_data{ptr, plane.length});
                }
            }
        }

    public:
        libcamera_capture()
            : running(false), controls(libcamera::controls::controls), stream(nullptr), fps(30)
        {
        }

        void open(std::size_t camera_idx)
        {
            camera_manager = std::make_unique<libcamera::CameraManager>();
            int ret = camera_manager->start();
            if (ret)
                throw std::runtime_error("camera manager failed to start, code " + std::to_string(-ret));

            std::vector<std::shared_ptr<libcamera::Camera>> cameras = camera_manager->cameras();
            // Do not show USB webcams as these are not supported in libcamera-apps!
            auto rem = std::remove_if(cameras.begin(), cameras.end(),
                                      [](auto &cam)
                                      { return cam->id().find("/usb") != std::string::npos; });
            cameras.erase(rem, cameras.end());

            if (cameras.size() == 0)
                throw std::runtime_error("no cameras available");

            if (camera_idx >= cameras.size())
                throw std::runtime_error("selected camera is not available");

            const std::string &cam_id = cameras[camera_idx]->id();
            camera = camera_manager->get(cam_id);
            if (!camera)
                throw std::runtime_error("Failed to find camera " + cam_id);

            if (camera->acquire())
                throw std::runtime_error("Failed to acquire camera " + cam_id);

            allocator = std::make_unique<libcamera::FrameBufferAllocator>(camera);
        }

        void configure(const stream_configuration &cfg)
        {
            config = camera->generateConfiguration({libcamera::StreamRole::VideoRecording});

            libcamera::StreamConfiguration &stream_config = config->at(0);

            stream_config.pixelFormat = conv_format(cfg.format);
            stream_config.size.width = cfg.width;
            stream_config.size.height = cfg.height;
            stream_config.bufferCount = 6;

            config->validate();
            camera->configure(config.get());

            stream = stream_config.stream();

            set_framerate(cfg.fps);
        }

        void set_exposure_time(int32_t time_us)
        {
            controls.set(libcamera::controls::ExposureTime, time_us);
        }
        void set_gain(float value)
        {
            controls.set(libcamera::controls::AnalogueGain, value);
        }
        void set_color_gain(float value_r, float value_b)
        {
            controls.set(libcamera::controls::ColourGains, {value_r, value_b});
        }
        void set_brightness(float value)
        {
            controls.set(libcamera::controls::Brightness, value);
        }
        void set_contrast(float value)
        {
            controls.set(libcamera::controls::Contrast, value);
        }
        void set_sharpness(float value)
        {
            controls.set(libcamera::controls::Sharpness, value);
        }
        void set_auto_exposure(bool value)
        {
            controls.set(libcamera::controls::AeEnable, value);
        }
        int32_t get_framerate() const
        {
            return this->fps;
        }

        void start(std::function<void(const libcamera_capture::buffer &)> frame_received)
        {
            this->frame_received = frame_received;

            controls.set(libcamera::controls::draft::NoiseReductionMode, libcamera::controls::draft::NoiseReductionModeFast);

            allocate_buffers();

            set_default_control();

            make_requests();

            camera->requestCompleted.connect(this, &libcamera_capture::request_completed);

            if (camera->start(&controls))
            {
                throw std::runtime_error("Failed to start camera");
            }
            controls.clear();
            for (std::unique_ptr<libcamera::Request> &request : requests)
            {
                if (camera->queueRequest(request.get()) < 0)
                {
                    throw std::runtime_error("Failed to queue request");
                }
            }

            th.reset(new std::thread([this]()
                                     {
            running.store(true);

            while (running.load())
            {
                libcamera::Request *r = nullptr;

                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&]
                            { return !res.empty() || !running; });

                    if (!running)
                    {
                        break;
                    }
                    if (res.size() > 0)
                    {
                        r = res.front();
                        res.pop();
                    }
                }
                if (r != nullptr)
                {
                    // process_res(r);
                }
            } }));
        }

        void stop()
        {
            if (!running.load())
            {
                return;
            }

            running.store(false);
            cv.notify_all();
            if (th && th->joinable())
            {
                th->join();
            }
            th.reset();

            camera->stop();

            for (auto &p : mapped_buffer)
            {
                munmap(p.second.pointer, p.second.length);
            }
            mapped_buffer.clear();
            if (stream)
            {
                allocator->free(stream);
            }
            stream = nullptr;
        }

        void close()
        {
            allocator.reset();
            if (camera)
            {
                camera->release();
                camera.reset();
            }
            if (camera_manager)
            {
                camera_manager->stop();
            }
        }
#else
    public:
        libcamera_capture() {}

        void open(std::size_t camera_idx) {}
        void configure(const stream_configuration &cfg) {}
        void start(std::function<void(const libcamera_capture::buffer &)> frame_received) {}
        void stop() {}
        void close() {}

        void set_exposure_time(int32_t time_us) {}
        void set_gain(float value) {}
        void set_color_gain(float value_r, float value_b) {}
        void set_brightness(float value) {}
        void set_contrast(float value) {}
        void set_sharpness(float value) {}
        void set_auto_exposure(bool value) {}

        int32_t get_framerate() const { return 0; }
#endif
    };

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

        virtual void run() override
        {
            camera.reset(new libcamera_capture());
            camera->open(0);
            camera->configure(libcamera_capture::stream_configuration(format, width, height, fps));

            for (const auto& [op, value] : request_options)
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
                        img.set_format(image_format::Y8_UINT);
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
                        stream_fmt = stream_format::Y8;
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

            camera->start(std::bind(&libcamera_capture_node::frame_received, this, std::placeholders::_1));

            if (emitter_enabled)
            {
                std::system("raspi-gpio set 18 op dh");
                std::system("raspi-gpio set 13 op dh");
                std::system("raspi-gpio set 19 op dh");
                std::system("raspi-gpio set 26 op dh");
            }
        }

        virtual void stop() override
        {
            if (emitter_enabled)
            {
                std::system("raspi-gpio set 18 op dl");
                std::system("raspi-gpio set 13 op dl");
                std::system("raspi-gpio set 19 op dl");
                std::system("raspi-gpio set 26 op dl");
            }

            if (running.load())
            {
                running.store(false);
                cv.notify_one();
                if (th && th->joinable())
                {
                    th->join();
                }
            }

            camera->stop();
            camera->close();
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
