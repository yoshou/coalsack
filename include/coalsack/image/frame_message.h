#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/core/graph_message.h"
#include "coalsack/image/image.h"

namespace coalsack {

enum class stream_type {
  ANY,
  DEPTH,
  COLOR,
  INFRARED,
};

enum class stream_format {
  ANY,
  Z16,
  RGB8,
  BGR8,
  RGBA8,
  BGRA8,
  Y8,
  Y16,
  YUYV,
  UYVY,
};

class stream_profile {
  int index;
  stream_type type;
  stream_format format;
  int fps;
  int uid;

 public:
  stream_profile(stream_type type = stream_type::ANY, int index = -1,
                 stream_format format = stream_format::ANY, int fps = 0, int uid = 0)
      : index(index), type(type), format(format), fps(fps), uid(uid) {}

  int get_index() const { return index; }
  void set_index(int index) { this->index = index; }

  stream_type get_type() const { return type; }
  void set_type(stream_type type) { this->type = type; }

  stream_format get_format() const { return format; }
  void set_format(stream_format format) { this->format = format; }

  int get_fps() const { return fps; }
  void set_fps(int fps) { this->fps = fps; }

  int get_unique_id() const { return uid; }
  void set_unique_id(int uid) { this->uid = uid; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(index, type, format, fps, uid);
  }
};

class frame_message_base : public graph_message {
  using time_type = double;

 protected:
  time_type timestamp;
  uint64_t frame_number;
  std::shared_ptr<stream_profile> profile;
  std::unordered_map<std::string, graph_message_ptr> metadata;

 public:
  frame_message_base() : timestamp(), frame_number(0), profile(), metadata() {}

  time_type get_timestamp() const { return timestamp; }
  void set_timestamp(time_type value) { timestamp = value; }
  uint64_t get_frame_number() const { return frame_number; }
  void set_frame_number(uint64_t value) { frame_number = value; }
  std::shared_ptr<stream_profile> get_profile() const { return profile; }
  void set_profile(std::shared_ptr<stream_profile> profile) { this->profile = profile; }
  graph_message_ptr get_metadata(const std::string &name) const { return metadata.at(name); }
  template <typename U>
  std::shared_ptr<U> get_metadata(const std::string &name) const {
    return std::dynamic_pointer_cast<U>(metadata.at(name));
  }
  void set_metadata(const std::string &name, graph_message_ptr value) { metadata[name] = value; }
  void set_metadata(const frame_message_base &other) {
    for (const auto &[name, data] : other.metadata) {
      metadata[name] = data;
    }
  }
};

template <typename T>
class frame_message : public frame_message_base {
  using data_type = T;

  data_type data;

  template <typename>
  friend class frame_message;

 public:
  frame_message() : data() {}

  void set_data(const T &data) { this->data = data; }
  void set_data(T &&data) { this->data = std::move(data); }
  T &get_data() { return data; }
  const T &get_data() const { return data; }

  static std::string get_type() { return std::string(typeid(T).name()) + "_frame"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(data, timestamp, frame_number, profile, metadata);
  }
};

using blob = std::vector<uint8_t>;
using blob_frame_message = frame_message<blob>;

using image_frame_message = frame_message<image>;

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::frame_message_base, coalsack::graph_message)

COALSACK_REGISTER_MESSAGE(coalsack::blob_frame_message, coalsack::frame_message_base)

COALSACK_REGISTER_MESSAGE(coalsack::image_frame_message, coalsack::frame_message_base)
