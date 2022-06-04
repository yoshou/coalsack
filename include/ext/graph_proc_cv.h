#pragma once

#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "graph_proc_img.h"
#include "blob_detector.h"
#include <opencv2/features2d.hpp>

namespace coalsack
{
    class fast_blob_detector_node : public graph_node
    {
        graph_edge_ptr output;
        cv::SimpleBlobDetector::Params params;

    public:
        fast_blob_detector_node()
            : graph_node(), output(std::make_shared<graph_edge>(this)), params(cv::SimpleBlobDetector::Params())
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "fast_blob_detector_node";
        }

        const cv::SimpleBlobDetector::Params &get_parameters() const
        {
            return params;
        }
        cv::SimpleBlobDetector::Params &get_parameters()
        {
            return params;
        }
        void set_parameters(const cv::SimpleBlobDetector::Params &params)
        {
            this->params = params;
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
            {
                const auto &src_image = image_msg->get_data();
                int cv_type = convert_to_cv_type(src_image.get_format());

                cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type, (void *)src_image.get_data());

                const auto width = static_cast<std::size_t>(src_mat.size().width);
                const auto height = static_cast<std::size_t>(src_mat.size().height);

                std::vector<circle_t> keypoints;
                blob_detector detector(src_mat.data, width, height);
                detector.min_dist_between_blobs = params.minDistBetweenBlobs;
                detector.min_threshold = params.minThreshold;
                detector.max_threshold = params.maxThreshold;
                detector.step_threshold = params.thresholdStep;
                detector.min_area = params.minArea;
                detector.max_area = params.maxArea;
                detector.min_circularity = params.minCircularity;
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
}

CEREAL_REGISTER_TYPE(coalsack::fast_blob_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::fast_blob_detector_node)
