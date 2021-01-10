#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "graph_proc_img.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

    static void imshow(std::string name, cv::InputArray mat)
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

static int convert_to_cv_type(image_format format)
{
    switch (format)
    {
    case image_format::Y8_UINT:
        return CV_8UC1;
    case image_format::R8G8B8_UINT:
    case image_format::B8G8R8_UINT:
        return CV_8UC3;
    case image_format::R8G8B8A8_UINT:
    case image_format::B8G8R8A8_UINT:
        return CV_8UC4;
    default:
        throw std::runtime_error("Invalid image format");
    }
}

class image_transform_node : public graph_node
{
    graph_edge_ptr output;

public:
    image_transform_node()
        : graph_node()
        , output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual void transform(const image& src_image, image& dst_image) = 0;

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
        {
            const auto& src_image = image_msg->get_data();
            
            image dst_image;
            transform(src_image, dst_image);

            auto msg = std::make_shared<frame_message<image>>();

            msg->set_data(std::move(dst_image));
            msg->set_profile(image_msg->get_profile());
            msg->set_timestamp(image_msg->get_timestamp());
            msg->set_frame_number(image_msg->get_frame_number());

            output->send(msg);
        }
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {}
};

CEREAL_REGISTER_TYPE(image_transform_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, image_transform_node)

class threshold_node : public image_transform_node
{
    double thresh;
    double maxval;
    int thresh_type;

public:
    threshold_node()
        : image_transform_node()
        , thresh(0)
        , maxval(255)
        , thresh_type(cv::THRESH_BINARY)
    {}

    virtual std::string get_proc_name() const override
    {
        return "threshold_node";
    }

    double get_threshold() const
    {
        return thresh;
    }
    void set_threshold(double value)
    {
        thresh = value;
    }
    double get_max_value()
    {
        return maxval;
    }
    void set_max_value(double value)
    {
        maxval = value;
    }
    int get_threshold_type() const
    {
        return thresh_type;
    }
    void set_threshold_type(int value)
    {
        thresh_type = value;
    }

    virtual void transform(const image&src_image, image& dst_image) override
    {
        image binary_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(), src_image.get_stride(), src_image.get_metadata_size());
        memcpy(binary_image.get_metadata(), src_image.get_metadata(), src_image.get_metadata_size());
        binary_image.set_format(src_image.get_format());
        
        int cv_type = convert_to_cv_type(src_image.get_format());

        cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type, (void*)src_image.get_data());
        cv::Mat dst_mat((int)binary_image.get_height(), (int)binary_image.get_width(), cv_type, (void*)binary_image.get_data());

        cv::threshold(src_mat, dst_mat, thresh, maxval, thresh_type);

        dst_image = std::move(binary_image);
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(thresh, maxval, thresh_type);
    }
};

CEREAL_REGISTER_TYPE(threshold_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(image_transform_node, threshold_node)

#include <opencv2/features2d.hpp>

struct keypoint
{
    float pt_x;
    float pt_y;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(pt_x, pt_y, size, angle, response, octave, class_id);
    }
};

using keypoint_frame_message = frame_message<std::vector<keypoint>>;

CEREAL_REGISTER_TYPE(keypoint_frame_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_message, keypoint_frame_message)

class orb_detector_node : public graph_node
{
    graph_edge_ptr output;
    cv::Ptr<cv::ORB> detector;

public:
    orb_detector_node()
        : graph_node()
        , output(std::make_shared<graph_edge>(this))
        , detector(cv::ORB::create())
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "orb_detector_node";
    }

    cv::Ptr<cv::ORB> get_detector() const
    {
        return detector;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
        {
            const auto& src_image = image_msg->get_data();
            int cv_type = convert_to_cv_type(src_image.get_format());

            cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type, (void *)src_image.get_data());

            std::vector<cv::KeyPoint> kps;
            detector->detect(src_mat, kps);

            std::vector<keypoint> pts;
            for (auto& kp: kps)
            {
                keypoint pt;
                pt.pt_x = kp.pt.x;
                pt.pt_y = kp.pt.y;
                pt.size = kp.size;
                pt.angle = kp.angle;
                pt.response = kp.response;
                pt.octave = kp.octave;
                pt.class_id = kp.class_id;
                pts.push_back(pt);
            }

            auto msg = std::make_shared<keypoint_frame_message>();

            msg->set_data(std::move(pts));
            msg->set_profile(image_msg->get_profile());
            msg->set_timestamp(image_msg->get_timestamp());
            msg->set_frame_number(image_msg->get_frame_number());

            output->send(msg);
        }
    }

    template<typename Archive>
    void load(Archive& archive)
    {
        int max_features;
        double scale_factor;
        int n_levels;
        int edge_threshold;
        int first_level;
        int wta_k;
        int score_type;
        int patch_size;
        int fast_threshold;

        archive(max_features, scale_factor, n_levels, edge_threshold,
            first_level, wta_k, score_type, patch_size, fast_threshold);

        detector->setMaxFeatures(max_features);
        detector->setScaleFactor(scale_factor);
        detector->setNLevels(n_levels);
        detector->setEdgeThreshold(edge_threshold);
        detector->setFirstLevel(first_level);
        detector->setWTA_K(wta_k);
        detector->setScoreType(score_type);
        detector->setPatchSize(patch_size);
        detector->setFirstLevel(fast_threshold);
    }

    template<typename Archive>
    void save(Archive& archive) const
    {
        int max_features = detector->getMaxFeatures();
        double scale_factor = detector->getScaleFactor();
        int n_levels = detector->getNLevels();
        int edge_threshold = detector->getEdgeThreshold();
        int first_level = detector->getFirstLevel();
        int wta_k = detector->getWTA_K();
        int score_type = detector->getScoreType();
        int patch_size = detector->getPatchSize();
        int fast_threshold = detector->getFirstLevel();

        archive(max_features, scale_factor, n_levels, edge_threshold,
            first_level, wta_k, score_type, patch_size, fast_threshold);
    }
};

CEREAL_REGISTER_TYPE(orb_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, orb_detector_node)

class simple_blob_detector_node : public graph_node
{
    graph_edge_ptr output;
    cv::SimpleBlobDetector::Params params;

public:
    simple_blob_detector_node()
        : graph_node()
        , output(std::make_shared<graph_edge>(this))
        , params(cv::SimpleBlobDetector::Params())
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "simple_blob_detector_node";
    }

    const cv::SimpleBlobDetector::Params& get_parameters() const
    {
        return params;
    }
    cv::SimpleBlobDetector::Params& get_parameters()
    {
        return params;
    }
    void set_parameters(const cv::SimpleBlobDetector::Params& params)
    {
        this->params = params;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
        {
            const auto& src_image = image_msg->get_data();
            int cv_type = convert_to_cv_type(src_image.get_format());

            cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type, (void *)src_image.get_data());

            auto detector = cv::SimpleBlobDetector::create(params);
            std::vector<cv::KeyPoint> kps;
            detector->detect(src_mat, kps);

            std::vector<keypoint> pts;
            for (auto& kp: kps)
            {
                keypoint pt;
                pt.pt_x = kp.pt.x;
                pt.pt_y = kp.pt.y;
                pt.size = kp.size;
                pt.angle = kp.angle;
                pt.response = kp.response;
                pt.octave = kp.octave;
                pt.class_id = kp.class_id;
                pts.push_back(pt);
            }

            auto msg = std::make_shared<keypoint_frame_message>();

            msg->set_data(std::move(pts));
            msg->set_profile(image_msg->get_profile());
            msg->set_timestamp(image_msg->get_timestamp());
            msg->set_frame_number(image_msg->get_frame_number());

            output->send(msg);
        }
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(params.blobColor);
        archive(params.filterByArea);
        archive(params.filterByCircularity);
        archive(params.filterByColor);
        archive(params.filterByConvexity);
        archive(params.filterByInertia);
        archive(params.maxArea);
        archive(params.maxCircularity);
        archive(params.maxConvexity);
        archive(params.maxInertiaRatio);
        archive(params.maxThreshold);
        archive(params.minArea);
        archive(params.minCircularity);
        archive(params.minConvexity);
        archive(params.minDistBetweenBlobs);
        archive(params.minInertiaRatio);
        archive(params.minRepeatability);
        archive(params.minThreshold);
        archive(params.thresholdStep);
    }
};

CEREAL_REGISTER_TYPE(simple_blob_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, simple_blob_detector_node)
