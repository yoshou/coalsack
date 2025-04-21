#pragma once

#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "blob_detector.h"
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#else
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#endif

namespace coalsack
{
    class fast_blob_detector_node : public graph_node
    {
    public:
        struct blob_detector_params
        {
            double min_threshold;
            double max_threshold;
            double step_threshold;
            double min_area;
            double max_area;
            double min_circularity;
            double max_circularity;
            float min_dist_between_blobs;
            std::int32_t min_repeatability;

            blob_detector_params()
            {
                min_threshold = 50;
                max_threshold = 220;
                step_threshold = 10;
                min_area = 25;
                max_area = 5000;
                min_circularity = 0.8f;
                max_circularity = std::numeric_limits<float>::max();
                min_dist_between_blobs = 10;
                min_repeatability = 2;
            }
        };

    private:
        graph_edge_ptr output;
        blob_detector_params params;

    public:
        fast_blob_detector_node()
            : graph_node(), output(std::make_shared<graph_edge>(this)), params()
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "fast_blob_detector_node";
        }

        const blob_detector_params &get_parameters() const
        {
            return params;
        }
        blob_detector_params &get_parameters()
        {
            return params;
        }
        void set_parameters(const blob_detector_params &params)
        {
            this->params = params;
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
            {
                const auto &src_image = image_msg->get_data();
                const auto data = src_image.get_data();
                const auto width = src_image.get_width();
                const auto height = src_image.get_height();
                const auto stride = src_image.get_stride();

                std::vector<circle_t> keypoints;
                blob_detector detector(data, width, height, stride);
                detector.min_dist_between_blobs = params.min_dist_between_blobs;
                detector.min_threshold = params.min_threshold;
                detector.max_threshold = params.max_threshold;
                detector.step_threshold = params.step_threshold;
                detector.min_area = params.min_area;
                detector.max_area = params.max_area;
                detector.min_circularity = params.min_circularity;
                detector.detect(keypoints);

                std::vector<keypoint> pts;
                for (const auto &kp : keypoints)
                {
                    keypoint pt;
                    pt.pt_x = kp.p.x;
                    pt.pt_y = kp.p.y;
                    pt.size = kp.r * 2.0f;
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

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(params.min_threshold);
            archive(params.max_threshold);
            archive(params.step_threshold);
            archive(params.min_area);
            archive(params.max_area);
            archive(params.min_circularity);
            archive(params.max_circularity);
            archive(params.min_dist_between_blobs);
            archive(params.min_repeatability);
        }
    };

    class charuco_detector_node : public graph_node
    {
    private:
        graph_edge_ptr output;
        cv::Size board_size;
        float square_length;
        float marker_length;
        int dict_type;

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
        cv::aruco::DetectorParameters detector_params;
        cv::aruco::CharucoParameters charuco_params;
        cv::Ptr<cv::aruco::CharucoDetector> detector;
#else
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        cv::Ptr<cv::aruco::CharucoBoard> board;
        cv::Ptr<cv::aruco::DetectorParameters> params;
#endif

    public:
        charuco_detector_node()
            : graph_node(), output(std::make_shared<graph_edge>(this))
            , board_size(3, 5)
            , square_length(0.0575f)
            , marker_length(0.0575f * 0.75f)
            , dict_type(cv::aruco::DICT_4X4_250)
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
            , detector_params()
            , charuco_params()
#endif
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "charuco_detector_node";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
        }

        virtual void initialize() override
        {
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
            const auto dictionary = cv::aruco::getPredefinedDictionary(dict_type);
            const auto board = cv::aruco::CharucoBoard(board_size, square_length, marker_length, dictionary);
            detector = cv::makePtr<cv::aruco::CharucoDetector>(board, charuco_params, detector_params);
#else
            const auto dictionary = cv::aruco::getPredefinedDictionary(dict_type);
            board = cv::aruco::CharucoBoard::create(board_size.width, board_size.height, square_length, marker_length, dictionary);
            params = cv::aruco::DetectorParameters::create();
#endif
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            const auto start = std::chrono::system_clock::now();
            if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
            {
                std::vector<int> marker_ids;
                std::vector<std::vector<cv::Point2f>> marker_corners;
                std::vector<int> charuco_ids;
                std::vector<cv::Point2f> charuco_corners;

                const auto &src_image = image_msg->get_data();

                int cv_type = convert_to_cv_type(src_image.get_format());

                cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type, (void *)src_image.get_data(), src_image.get_stride());

                try
                {
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
                    detector->detectBoard(src_mat, charuco_corners, charuco_ids, marker_corners, marker_ids);
#else
                    cv::aruco::detectMarkers(src_mat, board->dictionary, marker_corners, marker_ids, params);
                    if (marker_ids.size() > 0)
                    {
                        cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, src_mat, board, charuco_corners, charuco_ids);
                    }
#endif
                }
                catch (const cv::Exception &e)
                {
                    spdlog::error(e.what());
                }

                std::vector<keypoint> pts;
                for (size_t i = 0; i < charuco_ids.size(); i++)
                {
                    keypoint pt;
                    pt.pt_x = charuco_corners[i].x;
                    pt.pt_y = charuco_corners[i].y;
                    pt.size = 0;
                    pt.class_id = charuco_ids[i];
                    pts.push_back(pt);
                }

                std::cout << pts.size() << std::endl;

                auto msg = std::make_shared<keypoint_frame_message>();

                msg->set_data(std::move(pts));
                msg->set_profile(image_msg->get_profile());
                msg->set_timestamp(image_msg->get_timestamp());
                msg->set_frame_number(image_msg->get_frame_number());

                const auto end = std::chrono::system_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << elapsed << std::endl;

                output->send(msg);
            }
        }
    };
}

CEREAL_REGISTER_TYPE(coalsack::fast_blob_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::fast_blob_detector_node)

CEREAL_REGISTER_TYPE(coalsack::charuco_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::charuco_detector_node)
