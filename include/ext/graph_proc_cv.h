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
                int cv_type = convert_to_cv_type(src_image.get_format());

                cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type, (void *)src_image.get_data());

                const auto width = static_cast<std::size_t>(src_mat.size().width);
                const auto height = static_cast<std::size_t>(src_mat.size().height);

                std::vector<circle_t> keypoints;
                blob_detector detector(src_mat.data, width, height);
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
}

CEREAL_REGISTER_TYPE(coalsack::fast_blob_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::fast_blob_detector_node)
