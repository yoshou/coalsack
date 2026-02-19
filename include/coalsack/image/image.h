#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace coalsack {
enum class image_format {
  ANY,
  Y8_UINT,
  Y16_UINT,
  R8G8B8_UINT,
  R8G8B8A8_UINT,
  B8G8R8_UINT,
  B8G8R8A8_UINT,
  Z16_UINT,
  YUY2,
  UYVY,
};

class image {
 public:
  image() : data(), width(0), height(0), bpp(0), stride(0), format(image_format::ANY) {}

  image(uint32_t width, uint32_t height, uint32_t bpp, uint32_t stride)
      : data(stride * height),
        width(width),
        height(height),
        bpp(bpp),
        stride(stride),
        format(image_format::ANY) {}

  image(uint32_t width, uint32_t height, uint32_t bpp, uint32_t stride, const uint8_t *data)
      : image(width, height, bpp, stride) {
    std::copy_n(data, stride * height, this->data.begin());
  }

  image(const image &other)
      : data(other.data),
        width(other.width),
        height(other.height),
        bpp(other.bpp),
        stride(other.stride),
        format(other.format) {}

  image(image &&other)
      : data(std::move(other.data)),
        width(other.width),
        height(other.height),
        bpp(other.bpp),
        stride(other.stride),
        format(other.format) {}

  image &operator=(const image &other) {
    data = other.data;
    width = other.width;
    height = other.height;
    bpp = other.bpp;
    stride = other.stride;
    format = other.format;
    return *this;
  }

  image &operator=(image &&other) {
    data = std::move(other.data);
    width = other.width;
    height = other.height;
    bpp = other.bpp;
    stride = other.stride;
    format = other.format;
    return *this;
  }

  uint32_t get_width() const { return width; }
  uint32_t get_height() const { return height; }
  uint32_t get_bpp() const { return bpp; }
  uint32_t get_stride() const { return stride; }

  const uint8_t *get_data() const { return data.data(); }
  uint8_t *get_data() { return data.data(); }
  image_format get_format() const { return format; }
  void set_format(image_format format) { this->format = format; }

  bool empty() const { return data.empty(); }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(data, width, height, bpp, stride, format);
  }

 private:
  std::vector<uint8_t> data;
  uint32_t width;
  uint32_t height;
  uint32_t bpp;
  uint32_t stride;
  image_format format;
};
}  // namespace coalsack
