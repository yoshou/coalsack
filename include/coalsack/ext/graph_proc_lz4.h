/// @file graph_proc_lz4.h
/// @brief LZ4 compress/decompress nodes.
/// @ingroup ext_nodes
#pragma once

#include <lz4.h>

#include <vector>

#include "coalsack/image/frame_message.h"
#include "coalsack/image/image.h"
#include "coalsack/image/image_message.h"
#include "coalsack/image/image_nodes.h"

namespace coalsack {
struct lz4_image_header {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t bpp;
  std::uint32_t stride;
  image_format format;
};

/// @brief Compresses incoming @c frame_message<image> frames with LZ4.
/// @details Prepends a 16-byte @c lz4_image_header (width, height, stride, bpp, format) to
///          the raw pixel data, then compresses the whole block with @c LZ4_compress_default.
/// @par Inputs
/// - @b "default" — @c frame_message<image>
/// @par Outputs
/// - @b "default" — @c frame_message<blob> (LZ4 frame with header)
/// @par Properties
///   (none — compression level uses LZ4 defaults; not configurable at runtime)
/// @see decode_lz4_node, encode_jpeg_node
class encode_lz4_node : public graph_node {
  graph_edge_ptr output;

 public:
  encode_lz4_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual ~encode_lz4_node() {}

  virtual std::string get_proc_name() const override { return "encode_lz4"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
      const auto &image = image_msg->get_data();
      const auto width = image.get_width();
      const auto height = image.get_height();
      const auto bpp = image.get_bpp();
      const auto stride = image.get_stride();
      const auto format = image.get_format();

      const auto src_buf = image.get_data();
      const auto src_size = stride * height;
      const auto max_dst_size = LZ4_compressBound(src_size);
      const auto header_size = sizeof(lz4_image_header);
      std::vector<uint8_t> dst_buf(max_dst_size + header_size);
      const auto compressed_size = LZ4_compress_default(
          (const char *)src_buf, (char *)dst_buf.data() + header_size, src_size, max_dst_size);
      if (compressed_size <= 0) {
        spdlog::error("Failed to compress LZ4");
        return;
      }
      dst_buf.resize(compressed_size + header_size);

      const auto header = reinterpret_cast<lz4_image_header *>(dst_buf.data());
      header->width = width;
      header->height = height;
      header->stride = stride;
      header->bpp = bpp;
      header->format = format;

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

/// @brief Decompresses LZ4-encoded @c frame_message<blob> back to @c frame_message<image>.
/// @details Reads the 16-byte @c lz4_image_header to reconstruct the image geometry and
///          format, then decompresses the payload with @c LZ4_decompress_safe.
/// @par Inputs
/// - @b "default" — @c frame_message<blob> (LZ4 frame with header)
/// @par Outputs
/// - @b "default" — @c frame_message<image>
/// @par Properties
///   (none — image format is recovered from the LZ4 frame header)
/// @see encode_lz4_node
class decode_lz4_node : public graph_node {
  graph_edge_ptr output;

 public:
  decode_lz4_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual ~decode_lz4_node() {}

  virtual std::string get_proc_name() const override { return "decode_lz4"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto image_msg = std::dynamic_pointer_cast<frame_message<blob>>(message)) {
      const auto src_buf = image_msg->get_data().data();
      const auto src_size = image_msg->get_data().size();
      const auto header_size = sizeof(lz4_image_header);
      const auto header = reinterpret_cast<lz4_image_header *>(src_buf);
      const auto width = header->width;
      const auto height = header->height;
      const auto bpp = header->bpp;
      const auto stride = header->stride;
      const auto format = header->format;
      const auto dst_size = stride * height;

      image img(width, height, bpp, stride);
      img.set_format(format);

      const int decompressed_size =
          LZ4_decompress_safe((const char *)src_buf + header_size, (char *)img.get_data(),
                              src_size - header_size, dst_size);
      if (decompressed_size < 0) {
        spdlog::error("Failed to decompress LZ4");
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

COALSACK_REGISTER_NODE(coalsack::encode_lz4_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::decode_lz4_node, coalsack::graph_node)
