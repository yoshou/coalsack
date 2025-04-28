#pragma once

#include <jpeglib.h>

#include <atomic>
#include <map>
#include <thread>
#include <vector>

#include "graph_proc_img.h"

namespace coalsack {
class encode_jpeg_node : public graph_node {
  graph_edge_ptr output;

  struct jpeg_error_mgr jerr;
  struct jpeg_compress_struct cinfo;

  JSAMPROW row_pointer[1];
  std::vector<uint8_t> row_buffer;

  static void convert_yuyv_to_yuv(const uint8_t *src_buf, size_t width, uint8_t *dst_buf) {
    for (size_t i = 0; i < width; i += 2) {
      dst_buf[i * 3] = src_buf[i * 2 + 0];      // Y
      dst_buf[i * 3 + 1] = src_buf[i * 2 + 1];  // U
      dst_buf[i * 3 + 2] = src_buf[i * 2 + 3];  // V
      dst_buf[i * 3 + 3] = src_buf[i * 2 + 2];  // Y
      dst_buf[i * 3 + 4] = src_buf[i * 2 + 1];  // U
      dst_buf[i * 3 + 5] = src_buf[i * 2 + 3];  // V
    }
  }

  static void convert_uyvy_to_yuv(const uint8_t *src_buf, size_t width, uint8_t *dst_buf) {
    for (size_t i = 0; i < width; i += 2) {
      dst_buf[i * 3] = src_buf[i * 2 + 1];      // Y
      dst_buf[i * 3 + 1] = src_buf[i * 2 + 0];  // U
      dst_buf[i * 3 + 2] = src_buf[i * 2 + 2];  // V
      dst_buf[i * 3 + 3] = src_buf[i * 2 + 3];  // Y
      dst_buf[i * 3 + 4] = src_buf[i * 2 + 0];  // U
      dst_buf[i * 3 + 5] = src_buf[i * 2 + 2];  // V
    }
  }

  static void convert_bgr_to_rgb(const uint8_t *src_buf, size_t width, uint8_t *dst_buf) {
    for (size_t i = 0; i < width * 3; i += 3) {
      dst_buf[i] = src_buf[i + 2];      // R
      dst_buf[i + 1] = src_buf[i + 1];  // G
      dst_buf[i + 2] = src_buf[i];      // B
    }
  }

 public:
  encode_jpeg_node() : graph_node(), output(std::make_shared<graph_edge>(this)), row_buffer() {
    set_output(output);

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
  }

  virtual ~encode_jpeg_node() { jpeg_destroy_compress(&cinfo); }

  virtual std::string get_proc_name() const override { return "encode_jpeg"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      auto &image = image_msg->get_data();
      uint32_t width = image.get_width();
      uint32_t height = image.get_height();
      auto format = image_msg->get_profile()->get_format();

      auto src_buf = image.get_data();
      auto row_stride = (size_t)image.get_stride();

      cinfo.input_components = image.get_bpp();
      if (format == stream_format::YUYV || format == stream_format::UYVY) {
        cinfo.in_color_space = JCS_YCbCr;
        cinfo.input_components = 3;
      } else if (format == stream_format::Y8) {
        cinfo.in_color_space = JCS_GRAYSCALE;
        cinfo.input_components = 1;
      } else if (format == stream_format::RGB8 || format == stream_format::BGR8) {
        cinfo.in_color_space = JCS_RGB;
      } else {
        spdlog::error("Unsupported format");
        return;
      }

      const size_t row_size = cinfo.input_components * width;
      if (row_size > row_buffer.size()) {
        row_buffer.resize(row_size);
      }
      jpeg_set_defaults(&cinfo);

      unsigned long compressed_size = 0;
      unsigned char *data = nullptr;
      jpeg_mem_dest(&cinfo, &data, &compressed_size);

      cinfo.image_width = width;
      cinfo.image_height = height;

      jpeg_start_compress(&cinfo, TRUE);
      while (cinfo.next_scanline < cinfo.image_height) {
        if (format == stream_format::RGB8 || format == stream_format::Y8) {
          row_pointer[0] = &src_buf[cinfo.next_scanline * row_stride];
        } else if (format == stream_format::YUYV) {
          convert_yuyv_to_yuv(src_buf, width, &row_buffer[0]);
          row_pointer[0] = &row_buffer[0];
          src_buf += row_stride;
        } else if (format == stream_format::UYVY) {
          convert_uyvy_to_yuv(src_buf, width, &row_buffer[0]);
          row_pointer[0] = &row_buffer[0];
          src_buf += row_stride;
        } else if (format == stream_format::BGR8) {
          convert_bgr_to_rgb(src_buf, width, &row_buffer[0]);
          row_pointer[0] = &row_buffer[0];
          src_buf += row_stride;
        } else {
          spdlog::error("Failed to compress JPEG");
          return;
        }
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
      }
      jpeg_finish_compress(&cinfo);

      std::vector<uint8_t> dst_buf(data, data + compressed_size);
      free(data);

      auto msg = std::make_shared<frame_message<blob>>();

      msg->set_data(std::move(dst_buf));
      msg->set_profile(image_msg->get_profile());
      msg->set_timestamp(image_msg->get_timestamp());
      msg->set_frame_number(image_msg->get_frame_number());

      output->send(msg);
    }
  }

  template <typename Archive>
  void serialize(Archive &archive) {}
};

class decode_jpeg_node : public graph_node {
  graph_edge_ptr output;

  struct jpeg_error_mgr jerr;
  struct jpeg_decompress_struct dinfo;
  std::vector<uint8_t> row_buffer;

  static void convert_yuv_to_yuyv(const uint8_t *src_buf, size_t width, uint8_t *dst_buf) {
    for (size_t i = 0; i < width; i += 2) {
      dst_buf[i * 2] = src_buf[i * 3];          // Y
      dst_buf[i * 2 + 1] = src_buf[i * 3 + 1];  // U
      dst_buf[i * 2 + 2] = src_buf[i * 3 + 3];  // Y
      dst_buf[i * 2 + 3] = src_buf[i * 3 + 2];  // V
    }
  }

  static void convert_yuv_to_uyvy(const uint8_t *src_buf, size_t width, uint8_t *dst_buf) {
    for (size_t i = 0; i < width; i += 2) {
      dst_buf[i * 2] = src_buf[i * 3 + 1];      // U
      dst_buf[i * 2 + 1] = src_buf[i * 3 + 0];  // Y
      dst_buf[i * 2 + 2] = src_buf[i * 3 + 2];  // V
      dst_buf[i * 2 + 3] = src_buf[i * 3 + 3];  // Y
    }
  }

  static void convert_rgb_to_bgr(const uint8_t *src_buf, size_t width, uint8_t *dst_buf) {
    for (size_t i = 0; i < width * 3; i += 3) {
      dst_buf[i] = src_buf[i + 2];      // B
      dst_buf[i + 1] = src_buf[i + 1];  // G
      dst_buf[i + 2] = src_buf[i];      // R
    }
  }

 public:
  decode_jpeg_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);

    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);
  }

  virtual ~decode_jpeg_node() { jpeg_destroy_decompress(&dinfo); }

  virtual std::string get_proc_name() const override { return "decode_jpeg"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<blob>>(message)) {
      auto src_buf = image_msg->get_data().data();
      auto src_size = image_msg->get_data().size();
      auto format = image_msg->get_profile()->get_format();

      uint32_t res{};
      jpeg_mem_src(&dinfo, src_buf, src_size);
      res = jpeg_read_header(&dinfo, TRUE);
      if (!res) {
        spdlog::error("Invalid header");
        return;
      }
      uint32_t width = dinfo.image_width;
      uint32_t height = dinfo.image_height;
      int bpp = dinfo.num_components;

      image_format image_format = image_format::ANY;
      switch (format) {
        case stream_format::RGB8:
          image_format = image_format::R8G8B8_UINT;
          dinfo.out_color_space = JCS_RGB;
          break;
        case stream_format::BGR8:
          image_format = image_format::B8G8R8_UINT;
          dinfo.out_color_space = JCS_RGB;
          break;
        case stream_format::YUYV:
          image_format = image_format::YUY2;
          dinfo.out_color_space = JCS_YCbCr;
          break;
        case stream_format::UYVY:
          image_format = image_format::UYVY;
          dinfo.out_color_space = JCS_YCbCr;
          break;
        case stream_format::Y8:
          image_format = image_format::Y8_UINT;
          dinfo.out_color_space = JCS_GRAYSCALE;
          break;
        default:
          spdlog::error("Unsupported format");
          return;
      }

      res = jpeg_start_decompress(&dinfo);
      if (!res) {
        spdlog::error("Failed to decompress JPEG");
        return;
      }
      uint32_t row_stride = width * dinfo.output_components;
      image img(width, height, bpp, row_stride);
      img.set_format(image_format);

      auto dst_buf = img.get_data();
      uint8_t *ptr = dst_buf;
      if (row_stride > row_buffer.size()) {
        row_buffer.resize(row_stride);
      }
      uint8_t *row_pointer = row_buffer.data();

      while (dinfo.output_scanline < dinfo.output_height) {
        int num_lines = jpeg_read_scanlines(&dinfo, &row_pointer, 1);
        if (num_lines <= 0) {
          spdlog::error("Failed to decompress JPEG");
          return;
        }
        if (format == stream_format::RGB8 || format == stream_format::Y8) {
          memcpy(ptr, &row_buffer[0], row_stride);
          ptr += row_stride;
        } else if (format == stream_format::YUYV) {
          convert_yuv_to_yuyv(&row_buffer[0], width, ptr);
          ptr += row_stride;
        } else if (format == stream_format::UYVY) {
          convert_yuv_to_uyvy(&row_buffer[0], width, ptr);
          ptr += row_stride;
        } else if (format == stream_format::BGR8) {
          convert_rgb_to_bgr(&row_buffer[0], width, ptr);
          ptr += row_stride;
        }
      }
      res = jpeg_finish_decompress(&dinfo);
      if (!res) {
        spdlog::error("Failed to decompress JPEG");
        return;
      }

      auto msg = std::make_shared<frame_message<image>>();

      msg->set_data(std::move(img));
      msg->set_profile(image_msg->get_profile());
      msg->set_timestamp(image_msg->get_timestamp());
      msg->set_frame_number(image_msg->get_frame_number());

      output->send(msg);
    }
  }

  template <typename Archive>
  void serialize(Archive &archive) {}
};
}  // namespace coalsack

CEREAL_REGISTER_TYPE(coalsack::encode_jpeg_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::encode_jpeg_node)

CEREAL_REGISTER_TYPE(coalsack::decode_jpeg_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::decode_jpeg_node)
