#pragma once

#include <vector>
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

#include <spdlog/spdlog.h>

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
#include <opencv2/imgproc/imgproc.hpp>

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

        enum class stream_format
        {
            YUV420,
            RGB888,
            SBGGR10,
        };

        struct stream_configuration
        {
            stream_configuration(stream_format format, uint32_t width = 0, uint32_t height = 0, uint32_t fps = 0)
                : format(format), width(width), height(height), fps(fps)
            {
            }

            stream_format format;
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
        std::mutex camera_stop_mutex;

        std::unique_ptr<libcamera::CameraConfiguration> config;
        libcamera::Stream *stream;
        libcamera::ControlList controls;
        std::vector<std::unique_ptr<libcamera::Request>> requests;
        std::unordered_map<int, mapped_buffer_data> mapped_buffer;
        double duration;

        std::chrono::system_clock::time_point system_clock_start;
        std::chrono::steady_clock::time_point steady_clock_start;

        std::function<void(const libcamera_capture::buffer &)> frame_received;

        void request_completed(libcamera::Request *request)
        {
            if (request->status() == libcamera::Request::RequestCancelled)
            {
                return;
            }

            std::lock_guard<std::mutex> lock(mtx);
            res.push(request);
            cv.notify_all();
            // process_res(request);
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

                const auto timestamp = (std::chrono::duration_cast<std::chrono::nanoseconds>(system_clock_start.time_since_epoch()).count() +
                                        (metadata.timestamp -
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(steady_clock_start.time_since_epoch()).count())) /
                                       1000000.0;

                // const double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                const std::uint32_t sequence = metadata.sequence;

                cv::Mat frame;
                if (cfg.pixelFormat == libcamera::formats::YUV420)
                {
                    std::vector<cv::Mat> planes;

                    void *y_ptr = mapped_buffer.at(buffer->planes().at(0).fd.get()).pointer;
                    void *u_ptr = mapped_buffer.at(buffer->planes().at(1).fd.get()).pointer;
                    void *v_ptr = mapped_buffer.at(buffer->planes().at(2).fd.get()).pointer;

                    frame = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC1, y_ptr, cfg.stride).clone();
                }
                else if (cfg.pixelFormat == libcamera::formats::RGB888)
                {
                    std::vector<cv::Mat> planes;

                    void *ptr = mapped_buffer.at(buffer->planes().at(0).fd.get()).pointer;
                    frame = cv::Mat(cfg.size.height, cfg.size.width, CV_8UC3, ptr, cfg.stride).clone();
                }
                else if (cfg.pixelFormat == libcamera::formats::SBGGR10)
                {
                    std::vector<cv::Mat> planes;

                    void *ptr = mapped_buffer.at(buffer->planes().at(0).fd.get()).pointer;
                    frame = cv::Mat(cfg.size.height, cfg.size.width, CV_16UC1, ptr, cfg.stride).clone();
                }

                this->frame_received(libcamera_capture::buffer{frame, timestamp, sequence});
            }

            std::lock_guard<std::mutex> lock(camera_stop_mutex);

            if (!running)
            {
                return;
            }

            /* Re-queue the Request to the camera. */
            request->reuse(libcamera::Request::ReuseBuffers);

            const int64_t frame_time = static_cast<int64_t>(duration * 1000); // in us
            controls.set(libcamera::controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));

            spdlog::debug("Frame duration control: {0}", frame_time);

            request->controls() = controls;

            if (camera->queueRequest(request) < 0)
                throw std::runtime_error("failed to queue request");
        }

        static libcamera::PixelFormat conv_format(stream_format format)
        {
            switch (format)
            {
            case stream_format::RGB888:
                return libcamera::formats::RGB888;
            case stream_format::YUV420:
                return libcamera::formats::YUV420;
            case stream_format::SBGGR10:
                return libcamera::formats::SBGGR10;
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

                const std::vector<std::unique_ptr<libcamera::FrameBuffer>> &buffers = allocator->buffers(stream);
                for (unsigned int i = 0; i < buffers.size(); ++i)
                {
                    const std::unique_ptr<libcamera::FrameBuffer> &buffer = buffers[i];
                    std::size_t buffer_size = 0;
                    for (std::size_t plane_idx = 0; plane_idx < buffer->planes().size(); plane_idx++)
                    {
                        auto &plane = buffer->planes()[plane_idx];
                        buffer_size += plane.length;

                        if (plane_idx == buffer->planes().size() - 1 || plane.fd.get() != buffer->planes()[plane_idx + 1].fd.get())
                        {
                            void *ptr = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                            if (!ptr)
                            {
                                throw std::runtime_error("Failed to map the buffer");
                            }
                            assert(mapped_buffer.find(plane.fd.get()) == mapped_buffer.end());
                            mapped_buffer.emplace(plane.fd.get(), mapped_buffer_data{ptr, buffer_size});
                            buffer_size = 0;
                        }
                    }
                }
            }
        }

        void set_framerate(int32_t fps)
        {
            this->duration = 1000.0 / fps;

            const int64_t frame_time = static_cast<int64_t>(duration * 1000); // in us
            controls.set(libcamera::controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
        }

        void set_default_control()
        {
            const float roi_width = 0;
            const float roi_height = 0;
            const float roi_x = 0;
            const float roi_y = 0;

            if (!controls.get(libcamera::controls::ScalerCrop) && roi_width != 0 && roi_height != 0)
            {
                libcamera::Rectangle sensor_area = *camera->properties().get(libcamera::properties::ScalerCropMaximum);
                int x = roi_x * sensor_area.width;
                int y = roi_y * sensor_area.height;
                int w = roi_width * sensor_area.width;
                int h = roi_height * sensor_area.height;
                libcamera::Rectangle crop(x, y, w, h);
                crop.translateBy(sensor_area.topLeft());
                controls.set(libcamera::controls::ScalerCrop, crop);
            }

            // Set the lens position to the default value
            if (camera->controls().count(&libcamera::controls::LensPosition) > 0)
            {
                const auto f = camera->controls().at(&libcamera::controls::LensPosition).def().get<float>();
                controls.set(libcamera::controls::LensPosition, f);
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
            }
        }

    public:
        libcamera_capture()
            : running(false), stream(nullptr), controls(libcamera::controls::controls), duration(1000.0 / 30)
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
            if (!config)
                throw std::runtime_error("failed to generate video configuration");

            libcamera::StreamConfiguration &stream_config = config->at(0);

            stream_config.pixelFormat = conv_format(cfg.format);
            stream_config.size.width = cfg.width;
            stream_config.size.height = cfg.height;
            stream_config.bufferCount = 6;

            libcamera::CameraConfiguration::Status validation = config->validate();
            if (validation == libcamera::CameraConfiguration::Invalid)
                throw std::runtime_error("failed to valid stream configurations");

            if (camera->configure(config.get()) < 0)
                throw std::runtime_error("failed to configure streams");

            stream = stream_config.stream();

            set_framerate(cfg.fps);
        }

        void set_interval(double time_ms)
        {
            this->duration = time_ms;
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
            controls.set(libcamera::controls::ColourGains, libcamera::Span<const float, 2>({value_r, value_b}));
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
            return static_cast<int32_t>(1000 / this->duration);
        }

        void start(std::function<void(const libcamera_capture::buffer &)> frame_received)
        {
            system_clock_start = std::chrono::system_clock::now();
            steady_clock_start = std::chrono::steady_clock::now();

            this->frame_received = frame_received;

            controls.set(libcamera::controls::draft::NoiseReductionMode, libcamera::controls::draft::NoiseReductionModeFast);

            allocate_buffers();

            set_default_control();

            make_requests();

            if (camera->start(&controls))
            {
                throw std::runtime_error("Failed to start camera");
            }
            controls.clear();
            running = true;

            camera->requestCompleted.connect(this, &libcamera_capture::request_completed);

            for (std::unique_ptr<libcamera::Request> &request : requests)
            {
                if (camera->queueRequest(request.get()) < 0)
                {
                    throw std::runtime_error("Failed to queue request");
                }
            }

            th.reset(new std::thread([this]()
                                     {
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
                    process_res(r);
                }
            } }));
        }

        void stop()
        {
            if (!running.load())
            {
                return;
            }
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                running = false;
            }
            {
                std::lock_guard<std::mutex> lock(camera_stop_mutex);

                if (camera->stop())
                    throw std::runtime_error("failed to stop camera");
            }

            cv.notify_all();
            if (th && th->joinable())
            {
                th->join();
            }
            th.reset();

            if (camera)
            {
                camera->requestCompleted.disconnect(this, &libcamera_capture::request_completed);
            }

            requests.clear();
            controls.clear();
            std::queue<libcamera::Request *>().swap(res);

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
            config.reset();
            allocator.reset();
            if (camera)
            {
                camera->release();
                camera.reset();
            }
            if (camera_manager)
            {
                camera_manager.reset();
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

        void set_interval(double interval) {}
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
}
