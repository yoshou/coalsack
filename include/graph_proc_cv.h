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

class video_viz_node : public graph_node
{
    std::string image_name;
    std::shared_ptr<frame_message<image>> image_msg;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;

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
            cv::namedWindow(image_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(image_name, cv::WINDOW_NORMAL, cv::WINDOW_NORMAL);

            while (running.load() && cv::waitKey(1))
            {
                if (image_msg)
                {
                    const auto& image = image_msg->get_data();
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
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
        {
            this->image_msg = image_msg;
        }
    }
};

CEREAL_REGISTER_TYPE(video_viz_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, video_viz_node)