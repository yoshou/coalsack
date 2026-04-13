/// @file graph_proc_cv.h
/// @brief OpenCV-based processing nodes: capture, visualization, detection, and transforms.
/// @ingroup image
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

#include "coalsack/image/frame_message.h"
#include "coalsack/image/image.h"
#include "coalsack/image/image_message.h"
#include "coalsack/image/image_nodes.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

#include "coalsack/image/imgproc.h"

namespace coalsack {
/// @brief Displays incoming @c image_message frames in an OpenCV @c imshow window.
/// @details Opens a named @c cv::namedWindow on @c run() and refreshes the display in a
///          background thread every 1 ms via @c cv::waitKey.  The window is destroyed on @c stop().
/// @par Inputs
/// - @b "default" — @c image_message
/// @par Outputs
///   (none)
/// @par Properties
/// - image_name (std::string, "") — window name passed to @c cv::namedWindow and @c cv::imshow
/// @see video_viz_node, image_write_node
/// @ingroup image
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

/// @brief Saves incoming image frames to disk using cv::imwrite.
/// @details Calls @c cv::imwrite on each arriving @c image_message, writing the frame data
///          to the configured output path.  The directory must exist before the node runs.
/// @par Inputs
/// - @b "default" — @c image_message
/// @par Outputs
///   (none)
/// @par Properties
/// - path (std::string, "") — file path (including extension) to write each image frame to
/// @see image_viz_node, image_write_node
/// @ingroup image
class image_write_node : public graph_node {
  std::string path;

 public:
  image_write_node() : graph_node() {}

  void set_path(std::string path) { this->path = path; }

  std::string get_path() const { return path; }

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

/// @brief Displays video frames in an OpenCV imshow window.
/// @details Buffers each incoming @c frame_message<image> or @c image_message and renders
///          it via @c cv_window::imshow, which batches display updates across all windows
///          sharing the same @c cv::waitKey event loop.
/// @par Inputs
/// - @b "default" — @c frame_message<image> or @c image_message (rendered via cv::imshow)
/// @par Outputs
///   (none — display-only sink)
/// @par Properties
/// - image_name (std::string, "") — OpenCV window name passed to cv::imshow
/// @see image_viz_node, video_capture_node
/// @ingroup image
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

/// @brief Base class for single-input single-output image transformation nodes (OpenCV-based).
/// @details Derived classes implement @c transform(src, dst); this base unwraps the incoming
///          @c frame_message, calls @c transform(), then re-wraps the result with the original
///          profile, timestamp, and frame number before sending it on @b "default".
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<image> (transformed)
/// @par Properties
///   (none — defined by derived class)
/// @see threshold_node, gaussian_blur_node, resize_node, mask_node, scale_node, scale_abs_node
/// @ingroup image
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

/// @brief Applies a threshold operation (cv::threshold) to the input image.
/// @details Converts the source @c frame_message<image> to an OpenCV @c cv::Mat and calls
///          @c cv::threshold with the configured threshold, max_value, and type parameters.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<image> with threshold applied
/// @par Properties
/// - threshold (double, 0.0) — threshold value passed to cv::threshold
/// - max_value (double, 255.0) — maximum output value
/// - threshold_type (int, cv::THRESH_BINARY) — OpenCV threshold type constant
/// @see image_transform_node, mask_node, gaussian_blur_node
/// @ingroup image
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

/// @brief Applies a binary mask to the input image (sets masked pixels to zero).
/// @details Accepts an optional secondary @b "mask" input port (@c image_message) to update
///          the mask at runtime.  Uses @c cv::bitwise_and; the mask is auto-resized to the
///          source dimensions when they differ.  If no mask has been set, the frame passes through.
/// @par Inputs
/// - @b "default" — @c frame_message<image> (source)
/// - @b "mask"    — @c image_message (optional, updates the binary mask)
/// @par Outputs
/// - @b "default" — @c frame_message<image> with the mask applied
/// @par Properties
///   (none — the mask is supplied via the @b "mask" input or @c set_mask())
/// @see image_transform_node, threshold_node, mask_generator_node
/// @ingroup image
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

    int cv_type = convert_to_cv_type(src_image.get_format());

    cv::Mat src_mat((int)src_image.get_height(), (int)src_image.get_width(), cv_type,
                    (void *)src_image.get_data(), (int)src_image.get_stride());

    // If mask is empty, pass through unchanged
    if (mask.get_width() == 0 || mask.get_height() == 0) {
      dst_image = src_image;
      return;
    }

    image masked_image(src_image.get_width(), src_image.get_height(), src_image.get_bpp(),
                       src_image.get_stride());
    masked_image.set_format(src_image.get_format());

    cv::Mat dst_mat((int)masked_image.get_height(), (int)masked_image.get_width(), cv_type,
                    (void *)masked_image.get_data(), (int)masked_image.get_stride());
    cv::Mat mask_mat((int)mask.get_height(), (int)mask.get_width(), CV_8UC1,
                     (void *)mask.get_data(), (int)mask.get_stride());

    // Resize mask if size differs from source
    if (mask_mat.size() != src_mat.size()) {
      cv::resize(mask_mat, mask_mat, src_mat.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::bitwise_and(src_mat, mask_mat, dst_mat);

    dst_image = std::move(masked_image);
  }
};

/// @brief Applies a Gaussian blur (cv::GaussianBlur) to the input image.
/// @details Converts the source frame to a @c cv::Mat and calls @c cv::GaussianBlur using
///          the configured kernel size and sigma values.  Kernel dimensions must be odd and positive.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<image> with Gaussian blur applied
/// @par Properties
/// - kernel_width (int, 1) — blur kernel width (must be odd and positive)
/// - kernel_height (int, 1) — blur kernel height (must be odd and positive)
/// - sigma_x (double, 1.0) — Gaussian standard deviation in X
/// - sigma_y (double, 1.0) — Gaussian standard deviation in Y
/// @see image_transform_node, resize_node, threshold_node
/// @ingroup image
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

/// @brief Resizes the input image to a fixed width/height using cv::resize.
/// @details Allocates an output buffer of the target dimensions and calls @c cv::resize with
///          the configured interpolation method.  Width and height of 0 pass the frame through unchanged.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<image> resized to the configured dimensions
/// @par Properties
/// - width (uint32_t, 0) — output image width in pixels (0 = pass through)
/// - height (uint32_t, 0) — output image height in pixels (0 = pass through)
/// - interpolation_type (int, cv::INTER_LINEAR) — OpenCV interpolation flag
/// @see image_transform_node, gaussian_blur_node
/// @ingroup image
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

/// @brief Applies cv::convertScaleAbs with configurable alpha and beta.
/// @details Calls @c cv::convertScaleAbs(src, dst, alpha, beta); the output is always 8-bit
///          unsigned (CV_8U) regardless of the source depth.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<image> with pixel values scaled to uint8
/// @par Properties
/// - alpha (double, 1.0) — scale factor applied to each pixel value
/// - beta (double, 0.0) — offset added after scaling
/// @see image_transform_node, scale_node
/// @ingroup image
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

/// @brief Scales pixel values of the input image by alpha and beta.
/// @details Performs per-pixel linear scaling: @c dst = alpha * src + beta using
///          @c cv::Mat::convertTo, preserving the original pixel depth.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<image> with pixel values scaled (preserving original depth)
/// @par Properties
/// - alpha (double, 1.0) — scale factor applied to each pixel value
/// - beta (double, 0.0) — offset added after scaling
/// @see image_transform_node, scale_abs_node
/// @ingroup image
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

/// @brief Detects ORB keypoints and extracts descriptors from the input image.
/// @details Initialises a @c cv::ORB detector (configurable via @c get_detector()) and calls
///          @c detect on each incoming frame, then wraps the keypoints in a @c keypoint_frame_message.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c keypoint_frame_message
/// @par Properties
/// - max_features (int, 500) — maximum number of keypoints to retain
/// - scale_factor (double, 1.2) — pyramid scale factor
/// - n_levels (int, 8) — number of pyramid levels
/// - edge_threshold (int, 31) — border size where features are not detected
/// - first_level (int, 0) — first pyramid level to use
/// - wta_k (int, 2) — number of points in each oriented BRIEF descriptor element
/// - patch_size (int, 31) — size of the patch used for ORB descriptor computation
/// - fast_threshold (int, 20) — FAST threshold for keypoint detection
/// @see simple_blob_detector_node, fast_blob_detector_node
/// @ingroup image
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

/// @brief Detects blob keypoints using cv::SimpleBlobDetector.
/// @details Constructs a new @c cv::SimpleBlobDetector from the stored parameters on each
///          incoming frame and emits detected blob positions as a @c keypoint_frame_message.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c keypoint_frame_message
/// @par Properties
/// - parameters (cv::SimpleBlobDetector::Params) — full blob filter parameter set;
///   configure via @c get_parameters() / @c set_parameters()
/// @see orb_detector_node, fast_blob_detector_node, detect_circle_grid_node
/// @ingroup image
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

/// @brief Detects a circle calibration grid (cv::findCirclesGrid) in the input image.
/// @details Builds a @c cv::SimpleBlobDetector from @c params, then calls
///          @c cv::findCirclesGrid with the configured row/column counts and flags.
///          Detected circle centers are emitted as a @c keypoint_frame_message.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c keypoint_frame_message with detected circle centers
/// @par Properties
/// - parameters (simple_blob_detector_params) — blob filter settings used for circle detection;
///   configure via @c get_parameters() / @c set_parameters()
/// - num_circles_per_row (int, 2) — expected number of circle columns in the grid
/// - num_circles_per_column (int, 9) — expected number of circle rows in the grid
/// - flags (int, CALIB_CB_ASYMMETRIC_GRID|CALIB_CB_CLUSTERING) — cv::findCirclesGrid flags
/// @see simple_blob_detector_node, charuco_detector_node
/// @ingroup image
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

/// @brief Captures frames from a camera or file via cv::VideoCapture.
/// @details Opens device index 0 with @c cv::VideoCapture, applies any configured
///          @c request_options (cv property id/value pairs), and emits each grabbed
///          frame as a @c frame_message<image> timestamped from @c CAP_PROP_POS_MSEC.
/// @par Inputs
///   (none — autonomous source)
/// @par Outputs
/// - @b "default" — @c frame_message<image>
/// @par Properties
/// - stream (stream_type, COLOR) — stream type tag attached to the output profile
/// - request_options (vector of (int, double)) — cv::VideoCapture property overrides;
///   add via the @c set_option() helper before @c run()
/// @see image_viz_node, video_viz_node
/// @ingroup image
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

COALSACK_REGISTER_NODE(coalsack::image_viz_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::image_write_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::video_viz_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::image_transform_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::threshold_node, coalsack::image_transform_node)

COALSACK_REGISTER_NODE(coalsack::mask_node, coalsack::image_transform_node)

COALSACK_REGISTER_NODE(coalsack::gaussian_blur_node, coalsack::image_transform_node)

COALSACK_REGISTER_NODE(coalsack::resize_node, coalsack::image_transform_node)

COALSACK_REGISTER_NODE(coalsack::scale_abs_node, coalsack::image_transform_node)

COALSACK_REGISTER_NODE(coalsack::scale_node, coalsack::image_transform_node)

COALSACK_REGISTER_MESSAGE(coalsack::keypoint_frame_message, coalsack::frame_message_base)

COALSACK_REGISTER_NODE(coalsack::orb_detector_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::simple_blob_detector_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::detect_circle_grid_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::video_capture_node, coalsack::graph_node)
