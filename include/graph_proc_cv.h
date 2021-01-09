#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "graph_proc_img.h"
#include <opencv2/highgui.hpp>

class image_viz_node: public graph_node
{
    std::string image_name;
    std::shared_ptr<image_message> image_msg;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;
public:
    image_viz_node()
        : graph_node()
        , th()
        , running(false)
    {
    }

    void set_image_name(std::string name)
    {
        this->image_name = name;
    }

    std::string get_image_name() const
    {
        return image_name;
    }
    
    virtual std::string get_proc_name() const override
    {
        return "image_viz";
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(image_name);
    }

    virtual void run() override
    {
        running = true;
        th.reset(new std::thread([&]() {
            cv::namedWindow(image_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(image_name, cv::WINDOW_NORMAL, cv::WINDOW_NORMAL);

            while (running.load() && cv::waitKey(1))
            {
                if (image_msg)
                {
                    const auto& image = image_msg->get_image();
                    cv::Mat frame(image.get_height(), image.get_width(), CV_8UC3);
                    frame.data = (uchar*)image.get_data();
                    cv::imshow(image_name, frame);
                }
            }

            cv::destroyWindow(image_name);
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

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto image_msg = std::dynamic_pointer_cast<image_message>(message))
        {
            this->image_msg = image_msg;
        }
    }
};

CEREAL_REGISTER_TYPE(image_viz_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, image_viz_node)

class image_write_node: public graph_node
{
    std::string path;
public:
    image_write_node()
        : graph_node()
    {
    }

    void set_path(std::string path)
    {
        this->path = path;
    }

    std::string get_image_name() const
    {
        return path;
    }
    
    virtual std::string get_proc_name() const override
    {
        return "image_write";
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(path);
    }

    virtual void run() override
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto image_msg = std::dynamic_pointer_cast<image_message>(message))
        {
            const auto& image = image_msg->get_image();
            cv::Mat frame(image.get_height(), image.get_width(), CV_8UC3);
            frame.data = (uchar*)image.get_data();
            cv::imwrite(path, frame);

            spdlog::debug("Saved image to '{0}'", path);
        }
    }
};

CEREAL_REGISTER_TYPE(image_write_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, image_write_node)

class cv_window
{
public:
    static std::mutex& get_mutex()
    {
        static std::mutex mtx;
        return mtx;
    }

    static int wait_key(int delay)
    {
        std::lock_guard<std::mutex> lock(cv_window::get_mutex());
        return cv::waitKey(1);
    }

    static void create_window(std::string name)
    {
        std::lock_guard<std::mutex> lock(cv_window::get_mutex());
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::setWindowProperty(name, cv::WINDOW_NORMAL, cv::WINDOW_NORMAL);
    }

    static void destroy_window(std::string name)
    {
        std::lock_guard<std::mutex> lock(cv_window::get_mutex());
        cv::destroyWindow(name);
    }

    static void imshow(std::string name, const cv::InputArray& mat)
    {
        cv::imshow(name, mat);
    }
};

class video_viz_node : public graph_node
{
    std::string image_name;
    std::shared_ptr<frame_message<image>> image_msg;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;
    std::mutex mtx;

public:
    video_viz_node()
        : graph_node()
        , th()
        , running(false)
    {
    }

    void set_image_name(std::string name)
    {
        this->image_name = name;
    }

    std::string get_image_name() const
    {
        return image_name;
    }

    virtual std::string get_proc_name() const override
    {
        return "video_viz";
    }

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(image_name);
    }

    virtual void run() override
    {
        running = true;
        th.reset(new std::thread([&]() {
            cv_window::create_window(image_name);

            while (running.load() && cv_window::wait_key(0))
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (image_msg)
                {
                    const auto& image = image_msg->get_data();

                    int type = -1;
                    if (image_msg->get_profile())
                    {
                        auto format = image_msg->get_profile()->get_format();
                        switch (format)
                        {
                        case stream_format::Y8:
                            type = CV_8UC1;
                            break;
                        case stream_format::RGB8:
                            type = CV_8UC3;
                            break;
                        case stream_format::RGBA8:
                            type = CV_8UC4;
                            break;
                        case stream_format::BGR8:
                            type = CV_8UC3;
                            break;
                        case stream_format::BGRA8:
                            type = CV_8UC4;
                            break;
                        default:
                            break;
                        }
                    }

                    if (type < 0)
                    {
                        throw std::logic_error("Unknown image format");
                    }

                    cv::Mat frame(image.get_height(), image.get_width(), type, (uchar *)image.get_data());
                    cv_window::imshow(image_name, frame);
                }
            }

            cv_window::destroy_window(image_name);
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

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
        {
            std::lock_guard<std::mutex> lock(mtx);
            this->image_msg = image_msg;
        }
    }
};

CEREAL_REGISTER_TYPE(video_viz_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, video_viz_node)
