#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_proc_img.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

#include "imgproc.h"

namespace coalsack {
class image_viz_node : public graph_node {
  std::string image_name;
  std::shared_ptr<image_message> image_msg;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  image_viz_node() : graph_node(), th(), running(false) {}

  void set_image_name(std::string name) { this->image_name = name; }

  std::string get_image_name() const { return image_name; }

  virtual std::string get_proc_name() const override { return "image_viz"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(image_name);
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([&]() {
      cv::namedWindow(image_name, cv::WINDOW_NORMAL);
      cv::setWindowProperty(image_name, cv::WINDOW_NORMAL, cv::WINDOW_NORMAL);

      while (running.load() && cv::waitKey(1)) {
        if (image_msg) {
          const auto &image = image_msg->get_image();
          cv::Mat frame(image.get_height(), image.get_width(), CV_8UC3, (uchar *)image.get_data(),
                        image.get_stride());
          cv::imshow(image_name, frame);
        }
      }

      cv::destroyWindow(image_name);
    }));
  }

  virtual void stop() override {
    if (running.load()) {
      running.store(false);
      if (th && th->joinable()) {
        th->join();
      }
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<image_message>(message)) {
      this->image_msg = image_msg;
    }
  }
};

class image_write_node : public graph_node {
  std::string path;

 public:
  image_write_node() : graph_node() {}

  void set_path(std::string path) { this->path = path; }

  std::string get_image_name() const { return path; }

  virtual std::string get_proc_name() const override { return "image_write"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(path);
  }

  virtual void run() override {}

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<image_message>(message)) {
      const auto &image = image_msg->get_image();
      cv::Mat frame(image.get_height(), image.get_width(), CV_8UC3, (uchar *)image.get_data(),
                    image.get_stride());
      cv::imwrite(path, frame);

      spdlog::debug("Saved image to '{0}'", path);
    }
  }
};

class cv_window {
 public:
  static std::map<std::string, cv::Mat> &get_window_buf() {
    static std::map<std::string, cv::Mat> window_buf;
    return window_buf;
  }
  static std::mutex &get_mutex() {
    static std::mutex mtx;
    return mtx;
  }

  static int wait_key(int delay) {
    std::map<std::string, cv::Mat> window_buf_copy;
    {
      std::lock_guard<std::mutex> lock(cv_window::get_mutex());
      auto &window_buf = get_window_buf();
      window_buf_copy = window_buf;
      window_buf.clear();
    }
    for (const auto &[name, buf] : window_buf_copy) {
      cv::imshow(name, buf);
    }

    return cv::waitKey(delay);
  }

  static void create_window(std::string name) {
    std::lock_guard<std::mutex> lock(cv_window::get_mutex());
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(name, cv::WINDOW_NORMAL, cv::WINDOW_NORMAL);
  }

  static void destroy_window(std::string name) {
    std::lock_guard<std::mutex> lock(cv_window::get_mutex());
    cv::destroyWindow(name);

    auto &window_buf = get_window_buf();
    window_buf.erase(name);
  }

  static void imshow(std::string name, cv::Mat mat) {
    std::lock_guard<std::mutex> lock(cv_window::get_mutex());

    auto &window_buf = get_window_buf();
    window_buf[name] = mat;
  }
};

static int stream_format_to_cv_type(stream_format format) {
  int type = -1;
  switch (format) {
    case stream_format::Y8:
      type = CV_8UC1;
      break;
    case stream_format::Y16:
      type = CV_16UC1;
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
    case stream_format::YUYV:
      type = CV_8UC2;
      break;
    default:
      break;
  }
  return type;
}

class video_viz_node : public graph_node {
  std::string image_name;

 public:
  video_viz_node() : graph_node() {}

  void set_image_name(std::string name) { this->image_name = name; }

  std::string get_image_name() const { return image_name; }

  virtual std::string get_proc_name() const override { return "video_viz"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(image_name);
  }

  virtual void run() override {}

  virtual void stop() override {}

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      const auto &image = image_msg->get_data();

      int type = -1;
      stream_format format = stream_format::ANY;
      if (image_msg->get_profile()) {
        format = image_msg->get_profile()->get_format();
        type = stream_format_to_cv_type(format);
      }

      if (type < 0) {
        throw std::logic_error("Unknown image format");
      }

      auto frame = cv::Mat(image.get_height(), image.get_width(), type, (uchar *)image.get_data(),
                           image.get_stride())
                       .clone();
      if (format == stream_format::YUYV) {
        cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUYV);
      }
      cv_window::imshow(image_name, frame);
    }
  }
};

static int convert_to_cv_type(image_format format) {
  switch (format) {
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

class image_transform_node : public graph_node {
  graph_edge_ptr output;

 public:
  image_transform_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual void transform(const image &src_image, image &dst_image) = 0;

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      const auto &src_image = image_msg->get_data();

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

  template <typename Archive>
  void serialize(Archive &archive) {}
};

class threshold_node : public image_transform_node {
  double thresh;
  double maxval;
  int thresh_type;

 public:
  threshold_node()
      : image_transform_node(), thresh(0), maxval(255), thresh_type(cv::THRESH_BINARY) {}

  virtual std::string get_proc_name() const override { return "threshold_node"; }

  double get_threshold() const { return thresh; }
  void set_threshold(double value) { thresh = value; }
  double get_max_value() { return maxval; }
  void set_max_value(double value) { maxval = value; }
  int get_threshold_type() const { return thresh_type; }
  void set_threshold_type(int value) { thresh_type = value; }

  virtual void transform(const image &src_image, image &dst_image) override {
    image binary_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(),
                       src_image.get_stride());
    binary_image.set_format(src_image.get_format());

    int cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), src_image.get_stride());
    cv::Mat dst_mat((int)binary_image.get_height(), (int)binary_image.get_width(), cv_type,
                    (void *)binary_image.get_data(), binary_image.get_stride());

    cv::threshold(src_mat, dst_mat, thresh, maxval, thresh_type);

    dst_image = std::move(binary_image);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(thresh, maxval, thresh_type);
  }
};

class mask_node : public image_transform_node {
  image mask;
  std::mutex mask_mutex;
  graph_edge_ptr output;

 public:
  mask_node() : image_transform_node(), mask() {}

  virtual std::string get_proc_name() const override { return "mask_node"; }

  void set_mask(const image &value) { mask = value; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(mask);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "mask") {
      if (auto image_msg = std::dynamic_pointer_cast<image_message>(message)) {
        std::lock_guard lock(mask_mutex);
        mask = image_msg->get_image();
      }

      return;
    }
    image_transform_node::process(input_name, message);
  }

  virtual void transform(const image &src_image, image &dst_image) override {
    std::lock_guard lock(mask_mutex);

    image masked_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(),
                       src_image.get_stride());
    masked_image.set_format(src_image.get_format());

    int cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), (int)src_image.get_stride());
    cv::Mat dst_mat((int)masked_image.get_height(), (int)masked_image.get_width(), cv_type,
                    (void *)masked_image.get_data(), (int)masked_image.get_stride());
    cv::Mat mask_mat((int)mask.get_height(), (int)mask.get_width(), CV_8UC1,
                     (void *)mask.get_data(), (int)mask.get_stride());

    cv::bitwise_and(src_mat, mask_mat, dst_mat);

    dst_image = std::move(masked_image);
  }
};

class gaussian_blur_node : public image_transform_node {
  int kernel_width;
  int kernel_height;
  double sigma_x;
  double sigma_y;

 public:
  gaussian_blur_node()
      : image_transform_node(), kernel_width(1), kernel_height(1), sigma_x(1.0), sigma_y(1.0) {}

  virtual std::string get_proc_name() const override { return "gaussian_blur_node"; }

  void set_kernel_width(int value) { kernel_width = value; }
  int get_kernel_width() const { return kernel_width; }
  void set_kernel_height(int value) { kernel_height = value; }
  int get_kernel_height() const { return kernel_height; }
  void set_sigma_x(double value) { sigma_x = value; }
  double get_sigma_x() const { return sigma_x; }
  void set_sigma_y(double value) { sigma_y = value; }
  double get_sigma_y() const { return sigma_y; }

  virtual void transform(const image &src_image, image &dst_image) override {
    image blurred_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(),
                        src_image.get_stride());
    blurred_image.set_format(src_image.get_format());

    int cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), src_image.get_stride());
    cv::Mat dst_mat((int)blurred_image.get_height(), (int)blurred_image.get_width(), cv_type,
                    (void *)blurred_image.get_data(), blurred_image.get_stride());

    if (dst_mat.channels() == 1) {
      gaussian_blur(src_mat, dst_mat, kernel_width, kernel_height, sigma_x, sigma_y);
    } else {
      cv::GaussianBlur(src_mat, dst_mat, cv::Size(kernel_width, kernel_height), sigma_x, sigma_y);
    }

    dst_image = std::move(blurred_image);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(kernel_width, kernel_height, sigma_x, sigma_y);
  }
};

class resize_node : public image_transform_node {
  std::uint32_t width;
  std::uint32_t height;
  int interpolation;

 public:
  resize_node() : width(0), height(0), interpolation(cv::INTER_LINEAR) {}

  virtual std::string get_proc_name() const override { return "resize_node"; }

  std::uint32_t get_width() const { return width; }
  void set_width(std::uint32_t value) { width = value; }
  std::uint32_t get_height() const { return height; }
  void set_height(std::uint32_t value) { height = value; }
  int get_interpolation_type() const { return interpolation; }
  void set_interpolation_type(int value) { interpolation = value; }

  virtual void transform(const image &src_image, image &dst_image) override {
    image resized_image(width, height, src_image.get_bpp(), width * src_image.get_bpp());
    resized_image.set_format(src_image.get_format());

    const auto cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), src_image.get_stride());
    cv::Mat dst_mat((int)resized_image.get_height(), (int)resized_image.get_width(), cv_type,
                    (void *)resized_image.get_data(), resized_image.get_stride());

    cv::resize(src_mat, dst_mat, dst_mat.size(), 0, 0, interpolation);

    dst_image = std::move(resized_image);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(width, height, interpolation);
  }
};

class scale_abs_node : public image_transform_node {
  double alpha;
  double beta;

 public:
  scale_abs_node() : image_transform_node(), alpha(1.0), beta(0.0) {}

  virtual std::string get_proc_name() const override { return "scale_abs_node"; }

  double get_alpha() const { return alpha; }
  void set_alpha(double value) { alpha = value; }
  double get_beta() const { return beta; }
  void set_beta(double value) { beta = value; }

  virtual void transform(const image &src_image, image &dst_image) override {
    image binary_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(),
                       src_image.get_stride());
    binary_image.set_format(src_image.get_format());

    int cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), src_image.get_stride());
    cv::Mat dst_mat((int)binary_image.get_height(), (int)binary_image.get_width(), cv_type,
                    (void *)binary_image.get_data(), binary_image.get_stride());

    cv::convertScaleAbs(src_mat, dst_mat, alpha, beta);

    dst_image = std::move(binary_image);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(alpha, beta);
  }
};

class scale_node : public image_transform_node {
  double alpha;
  double beta;

 public:
  scale_node() : image_transform_node(), alpha(1.0), beta(0.0) {}

  virtual std::string get_proc_name() const override { return "scale_node"; }

  double get_alpha() const { return alpha; }
  void set_alpha(double value) { alpha = value; }
  double get_beta() const { return beta; }
  void set_beta(double value) { beta = value; }

  virtual void transform(const image &src_image, image &dst_image) override {
    image binary_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(),
                       src_image.get_stride());
    binary_image.set_format(src_image.get_format());

    int cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), src_image.get_stride());
    cv::Mat dst_mat((int)binary_image.get_height(), (int)binary_image.get_width(), cv_type,
                    (void *)binary_image.get_data(), binary_image.get_stride());

    if (dst_mat.channels() == 1) {
      const auto stride = static_cast<std::size_t>(dst_mat.step);
      for (int y = 0; y < dst_mat.rows; y++) {
        const auto src_row = &src_mat.data[y * stride];
        const auto dst_row = &dst_mat.data[y * stride];
        int x = 0;
#if USE_NEON
        constexpr auto num_vector_lanes = static_cast<int>(sizeof(uint8x16_t) / sizeof(uint8_t));
        const auto v_alpha = vdupq_n_f32(static_cast<float>(alpha));
        const auto v_beta = vdupq_n_f32(static_cast<float>(beta));
        for (; x <= dst_mat.cols - num_vector_lanes; x += num_vector_lanes) {
          const uint8x16_t v_src = vld1q_u8(src_row + x);
          const uint16x8_t v_src_l = vmovl_u8(vget_low_u8(v_src));
          const uint16x8_t v_src_h = vmovl_u8(vget_high_u8(v_src));
          const uint32x4_t v_src0 = vmovl_u16(vget_low_u16(v_src_l));
          const uint32x4_t v_src1 = vmovl_u16(vget_high_u16(v_src_l));
          const uint32x4_t v_src2 = vmovl_u16(vget_low_u16(v_src_h));
          const uint32x4_t v_src3 = vmovl_u16(vget_high_u16(v_src_h));
          const uint32x4_t v_dst0 =
              vcvtq_u32_f32(vmlaq_f32(v_beta, v_alpha, vcvtq_f32_u32(v_src0)));
          const uint32x4_t v_dst1 =
              vcvtq_u32_f32(vmlaq_f32(v_beta, v_alpha, vcvtq_f32_u32(v_src1)));
          const uint32x4_t v_dst2 =
              vcvtq_u32_f32(vmlaq_f32(v_beta, v_alpha, vcvtq_f32_u32(v_src2)));
          const uint32x4_t v_dst3 =
              vcvtq_u32_f32(vmlaq_f32(v_beta, v_alpha, vcvtq_f32_u32(v_src3)));
          const uint16x8_t v_dst_l = vcombine_u16(vqmovn_u32(v_dst0), vqmovn_u32(v_dst1));
          const uint16x8_t v_dst_h = vcombine_u16(vqmovn_u32(v_dst2), vqmovn_u32(v_dst3));
          const uint8x16_t v_dst = vcombine_u8(vqmovn_u16(v_dst_l), vqmovn_u16(v_dst_h));
          vst1q_u8(dst_row + x, v_dst);
        }
#endif
        for (; x < dst_mat.cols; x++) {
          dst_row[x] =
              cv::saturate_cast<uchar>(static_cast<float>(alpha) * static_cast<float>(src_row[x]) +
                                       static_cast<float>(beta));
        }
      }
    } else if (dst_mat.channels() == 2) {
      for (int y = 0; y < dst_mat.rows; y++) {
        for (int x = 0; x < dst_mat.cols; x++) {
          for (int c = 0; c < dst_mat.channels(); c++) {
            dst_mat.at<cv::Vec2b>(y, x)[c] =
                cv::saturate_cast<uchar>(alpha * src_mat.at<cv::Vec2b>(y, x)[c] + beta);
          }
        }
      }
    } else if (dst_mat.channels() == 3) {
      for (int y = 0; y < dst_mat.rows; y++) {
        for (int x = 0; x < dst_mat.cols; x++) {
          for (int c = 0; c < dst_mat.channels(); c++) {
            dst_mat.at<cv::Vec3b>(y, x)[c] =
                cv::saturate_cast<uchar>(alpha * src_mat.at<cv::Vec3b>(y, x)[c] + beta);
          }
        }
      }
    }

    dst_image = std::move(binary_image);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(alpha, beta);
  }
};

struct keypoint {
  float pt_x;
  float pt_y;
  float size;
  float angle;
  float response;
  int octave;
  int class_id;

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(pt_x, pt_y, size, angle, response, octave, class_id);
  }
};

using keypoint_frame_message = frame_message<std::vector<keypoint>>;

class orb_detector_node : public graph_node {
  graph_edge_ptr output;
  cv::Ptr<cv::ORB> detector;

 public:
  orb_detector_node()
      : graph_node(), output(std::make_shared<graph_edge>(this)), detector(cv::ORB::create()) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "orb_detector_node"; }

  cv::Ptr<cv::ORB> get_detector() const { return detector; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      const auto &src_image = image_msg->get_data();
      int cv_type = convert_to_cv_type(src_image.get_format());

      cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                      (void *)src_image.get_data(), src_image.get_stride());

      std::vector<cv::KeyPoint> kps;
      detector->detect(src_mat, kps);

      std::vector<keypoint> pts;
      for (auto &kp : kps) {
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

  template <typename Archive>
  void load(Archive &archive) {
    int max_features;
    double scale_factor;
    int n_levels;
    int edge_threshold;
    int first_level;
    int wta_k;
    cv::ORB::ScoreType score_type;
    int patch_size;
    int fast_threshold;

    archive(max_features, scale_factor, n_levels, edge_threshold, first_level, wta_k, score_type,
            patch_size, fast_threshold);

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

  template <typename Archive>
  void save(Archive &archive) const {
    int max_features = detector->getMaxFeatures();
    double scale_factor = detector->getScaleFactor();
    int n_levels = detector->getNLevels();
    int edge_threshold = detector->getEdgeThreshold();
    int first_level = detector->getFirstLevel();
    int wta_k = detector->getWTA_K();
    cv::ORB::ScoreType score_type = detector->getScoreType();
    int patch_size = detector->getPatchSize();
    int fast_threshold = detector->getFirstLevel();

    archive(max_features, scale_factor, n_levels, edge_threshold, first_level, wta_k, score_type,
            patch_size, fast_threshold);
  }
};

class simple_blob_detector_node : public graph_node {
  graph_edge_ptr output;
  cv::SimpleBlobDetector::Params params;

 public:
  simple_blob_detector_node()
      : graph_node(),
        output(std::make_shared<graph_edge>(this)),
        params(cv::SimpleBlobDetector::Params()) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "simple_blob_detector_node"; }

  const cv::SimpleBlobDetector::Params &get_parameters() const { return params; }
  cv::SimpleBlobDetector::Params &get_parameters() { return params; }
  void set_parameters(const cv::SimpleBlobDetector::Params &params) { this->params = params; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      const auto &src_image = image_msg->get_data();
      int cv_type = convert_to_cv_type(src_image.get_format());

      cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                      (void *)src_image.get_data(), src_image.get_stride());

      auto detector = cv::SimpleBlobDetector::create(params);
      std::vector<cv::KeyPoint> kps;
      detector->detect(src_mat, kps);

      std::vector<keypoint> pts;
      for (auto &kp : kps) {
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

  template <typename Archive>
  void serialize(Archive &archive) {
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

struct simple_blob_detector_params {
  float threshold_step = 10;
  float min_threshold = 50;
  float max_threshold = 220;
  size_t min_repeatability = 2;
  float min_dist_between_blobs = 10;

  bool filter_by_color = true;
  uint8_t blob_color = 0;

  bool filter_by_area = true;
  float min_area = 25;
  float max_area = 5000;

  bool filter_by_circularity = false;
  float min_circularity = 0.8f;
  float max_circularity = std::numeric_limits<float>::max();

  bool filter_by_inertia = true;
  float min_inertia_ratio = 0.1f;
  float max_inertia_ratio = std::numeric_limits<float>::max();

  bool filter_by_convexity = true;
  float min_convexity = 0.95f;
  float max_convexity = std::numeric_limits<float>::max();

  bool collect_contours = false;

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(threshold_step, min_threshold, max_threshold, min_repeatability, min_dist_between_blobs,
            filter_by_color, blob_color, filter_by_area, min_area, max_area, filter_by_circularity,
            min_circularity, max_circularity, filter_by_inertia, min_inertia_ratio,
            max_inertia_ratio, filter_by_convexity, min_convexity, max_convexity, collect_contours);
  }
};

class detect_circle_grid_node : public graph_node {
  graph_edge_ptr output;
  simple_blob_detector_params params;
  int num_circles_per_row = 2;
  int num_circles_per_column = 9;
  int flags = cv::CALIB_CB_ASYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
  cv::Ptr<cv::SimpleBlobDetector> detector;

 public:
  detect_circle_grid_node()
      : graph_node(), output(std::make_shared<graph_edge>(this)), params(), detector() {
    set_output(output);
  }

  void set_parameters(const simple_blob_detector_params &params) { this->params = params; }
  const simple_blob_detector_params &get_parameters() const { return params; }
  simple_blob_detector_params &get_parameters() { return params; }

  void set_num_circles_per_row(int value) { num_circles_per_row = value; }
  int get_num_circles_per_row() const { return num_circles_per_row; }

  void set_num_circles_per_column(int value) { num_circles_per_column = value; }
  int get_num_circles_per_column() const { return num_circles_per_column; }

  void set_flags(int value) { flags = value; }
  int get_flags() const { return flags; }

  virtual std::string get_proc_name() const override { return "detect_circle_grid"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(params, num_circles_per_row, num_circles_per_column, flags);
  }

  virtual void initialize() override {
    auto params = cv::SimpleBlobDetector::Params();

    params.minThreshold = this->params.min_threshold;
    params.maxThreshold = this->params.max_threshold;
    params.thresholdStep = this->params.threshold_step;
    params.minRepeatability = this->params.min_repeatability;
    params.minDistBetweenBlobs = this->params.min_dist_between_blobs;
    params.filterByColor = this->params.filter_by_color;
    params.blobColor = this->params.blob_color;
    params.filterByArea = this->params.filter_by_area;
    params.minArea = this->params.min_area;
    params.maxArea = this->params.max_area;
    params.filterByCircularity = this->params.filter_by_circularity;
    params.minCircularity = this->params.min_circularity;
    params.maxCircularity = this->params.max_circularity;
    params.filterByInertia = this->params.filter_by_inertia;
    params.minInertiaRatio = this->params.min_inertia_ratio;
    params.maxInertiaRatio = this->params.max_inertia_ratio;
    params.filterByConvexity = this->params.filter_by_convexity;
    params.minConvexity = this->params.min_convexity;
    params.maxConvexity = this->params.max_convexity;

    detector = cv::SimpleBlobDetector::create(params);
  }

  virtual void finalize() override { detector.release(); }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      const auto &src_image = image_msg->get_data();
      cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(),
                      convert_to_cv_type(src_image.get_format()), (void *)src_image.get_data(),
                      src_image.get_stride());

      const cv::Size pattern_size(num_circles_per_row, num_circles_per_column);

      std::vector<cv::Point2f> centers;
      const auto found = cv::findCirclesGrid(src_mat, pattern_size, centers, flags, detector);

      if (!found) {
        centers.clear();
      }

      std::vector<keypoint> keypoints;
      for (const auto &corner : centers) {
        keypoint kp;
        kp.pt_x = corner.x;
        kp.pt_y = corner.y;
        kp.size = 0;
        kp.angle = 0;
        kp.response = 0;
        kp.octave = 0;
        kp.class_id = 0;
        keypoints.push_back(kp);
      }

      auto msg = std::make_shared<keypoint_frame_message>();
      msg->set_data(std::move(keypoints));
      msg->set_profile(image_msg->get_profile());
      msg->set_timestamp(image_msg->get_timestamp());
      msg->set_frame_number(image_msg->get_frame_number());

      output->send(msg);
    }
  }
};

class video_capture_node : public graph_node {
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;
  graph_edge_ptr output;
  std::vector<std::tuple<int, double>> request_options;
  int fps;
  stream_type stream;

 public:
  video_capture_node()
      : graph_node(),
        th(),
        running(false),
        output(std::make_shared<graph_edge>(this)),
        stream(stream_type::COLOR) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "video_capture"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(request_options);
    archive(stream);
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([&]() {
      cv::VideoCapture capture;
      capture.open(0);
      if (stream == stream_type::INFRARED) {
        capture.set(cv::CAP_PROP_CONVERT_RGB, 0);
      }
      for (const auto &[option, value] : request_options) {
        capture.set(option, value);
      }
      cv::Mat frame;

      if (!capture.isOpened()) {
        spdlog::error("Failed to open capture device");
        running = false;
      }

      fps = capture.get(cv::CAP_PROP_FPS);

      while (running.load()) {
        capture.read(frame);
        const auto timestamp = capture.get(cv::CAP_PROP_POS_MSEC);

        if (frame.empty()) {
          spdlog::error("Failed to grab frame");
          running = false;
        }

        video_frame_callback(frame, timestamp);
      }
    }));
  }

  void video_frame_callback(cv::Mat frame, double timestamp) {
    if (frame.empty()) {
      return;
    }
    if (stream == stream_type::INFRARED) {
      cv::extractChannel(frame, frame, 0);
    }

    auto msg = std::make_shared<frame_message<image>>();

    image img(static_cast<std::uint32_t>(frame.size().width),
              static_cast<std::uint32_t>(frame.size().height),
              static_cast<std::uint32_t>(frame.elemSize()), static_cast<std::uint32_t>(frame.step),
              (const uint8_t *)frame.data);

    if (frame.channels() == 1) {
      img.set_format(image_format::Y8_UINT);
    } else if (frame.channels() == 3) {
      img.set_format(image_format::B8G8R8_UINT);
    } else if (frame.channels() == 4) {
      img.set_format(image_format::B8G8R8A8_UINT);
    }
    stream_format stream_fmt = stream_format::ANY;
    if (frame.channels() == 1) {
      stream_fmt = stream_format::Y8;
    } else if (frame.channels() == 3) {
      stream_fmt = stream_format::BGR8;
    } else if (frame.channels() == 4) {
      stream_fmt = stream_format::BGRA8;
    }

    msg->set_data(std::move(img));
    msg->set_profile(std::make_shared<stream_profile>(stream, 0, stream_fmt, fps, 0));
    msg->set_timestamp(timestamp);
    msg->set_frame_number(0);

    output->send(msg);
  }

  virtual void stop() override {
    if (running.load()) {
      running.store(false);
      if (th && th->joinable()) {
        th->join();
      }
    }
  }

  void set_stream(stream_type stream) { this->stream = stream; }

  void set_option(int option, double value) {
    request_options.push_back(std::make_tuple(option, value));
  }
};
}  // namespace coalsack

CEREAL_REGISTER_TYPE(coalsack::image_viz_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::image_viz_node)

CEREAL_REGISTER_TYPE(coalsack::image_write_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::image_write_node)

CEREAL_REGISTER_TYPE(coalsack::video_viz_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::video_viz_node)

CEREAL_REGISTER_TYPE(coalsack::image_transform_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::image_transform_node)

CEREAL_REGISTER_TYPE(coalsack::threshold_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::image_transform_node, coalsack::threshold_node)

CEREAL_REGISTER_TYPE(coalsack::mask_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::image_transform_node, coalsack::mask_node)

CEREAL_REGISTER_TYPE(coalsack::gaussian_blur_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::image_transform_node, coalsack::gaussian_blur_node)

CEREAL_REGISTER_TYPE(coalsack::resize_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::image_transform_node, coalsack::resize_node)

CEREAL_REGISTER_TYPE(coalsack::scale_abs_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::image_transform_node, coalsack::scale_abs_node)

CEREAL_REGISTER_TYPE(coalsack::scale_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::image_transform_node, coalsack::scale_node)

CEREAL_REGISTER_TYPE(coalsack::keypoint_frame_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, coalsack::keypoint_frame_message)

CEREAL_REGISTER_TYPE(coalsack::orb_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::orb_detector_node)

CEREAL_REGISTER_TYPE(coalsack::simple_blob_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::simple_blob_detector_node)

CEREAL_REGISTER_TYPE(coalsack::detect_circle_grid_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::detect_circle_grid_node)

CEREAL_REGISTER_TYPE(coalsack::video_capture_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::video_capture_node)
