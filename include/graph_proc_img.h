#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_proc.h"
#include "syncer.h"

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

class image_message : public graph_message {
  image img;

 public:
  image_message() : img() {}

  void set_image(const image &img) { this->img = img; }
  void set_image(image &&img) { this->img = std::move(img); }
  const image &get_image() const { return img; }
  static std::string get_type() { return "image"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(img);
  }
};

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

class image_heartbeat_node : public heartbeat_node {
  image img;
  graph_edge_ptr output;

 public:
  image_heartbeat_node() : heartbeat_node(), img(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  void set_image(image img) { this->img = img; }

  const image &get_image() const { return img; }

  virtual std::string get_proc_name() const override { return "image_heartbeat"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(cereal::base_class<heartbeat_node>(this));
    archive(img);
  }

  virtual void tick() override {
    auto msg = std::make_shared<image_message>();
    msg->set_image(img);
    output->send(msg);
  }
};

struct approximate_time_sync_config {
  using sync_info = approximate_time;
  double interval;

  approximate_time_sync_config() : interval(0) {}

  explicit approximate_time_sync_config(double interval) : interval(interval) {}

  approximate_time create_sync_info(std::shared_ptr<frame_message_base> message) {
    return approximate_time(message->get_timestamp(), interval);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(interval);
  }

  void set_interval(double interval) { this->interval = interval; }
  double get_interval() const { return interval; }
};

struct frame_number_sync_config {
  using sync_info = frame_number;

  frame_number_sync_config() {}

  sync_info create_sync_info(std::shared_ptr<frame_message_base> message) {
    return frame_number(message->get_frame_number());
  }

  template <typename Archive>
  void serialize(Archive &archive) {}
};

template <typename Config = approximate_time_sync_config>
class sync_node : public graph_node {
  using sync_config = Config;
  using sync_info = typename sync_config::sync_info;
  using syncer_type = stream_syncer<graph_message_ptr, std::string, sync_info>;

  syncer_type syncer;
  graph_edge_ptr output;
  sync_config config;

 public:
  sync_node() : graph_node(), syncer(), output(std::make_shared<graph_edge>(this)), config() {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "sync_node"; }

  void set_config(sync_config config) { this->config = config; }
  const sync_config &get_config() const { return config; }
  sync_config &get_config() { return config; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(config);
  }

  virtual void run() override {
    syncer.start(std::make_shared<typename syncer_type::callback_type>(
        [this](const std::map<std::string, graph_message_ptr> &frames) {
          auto msg = std::make_shared<object_message>();
          for (auto frame : frames) {
            msg->add_field(frame.first, frame.second);
          }
          output->send(msg);
        }));
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      syncer.sync(input_name, frame_msg, config.create_sync_info(frame_msg));
    }
  }
};

using approximate_time_sync_node = sync_node<approximate_time_sync_config>;
using frame_number_sync_node = sync_node<frame_number_sync_config>;

class tiling_node : public graph_node {
  graph_edge_ptr output;
  std::uint32_t num_cols;
  std::uint32_t num_rows;
  std::set<std::string> names;

 public:
  tiling_node()
      : graph_node(), output(std::make_shared<graph_edge>(this)), num_cols(0), num_rows(0) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "tiling_node"; }

  void set_num_rows(std::uint32_t value) { num_rows = value; }
  void set_num_cols(std::uint32_t value) { num_cols = value; }
  std::uint32_t get_num_rows() const { return num_rows; }
  std::uint32_t get_num_cols() const { return num_cols; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(num_rows);
    archive(num_cols);
  }

  void tiling(const std::vector<const image *> &images, image &output) {
    if (images.size() == 0) {
      return;
    }
    uint32_t tile_width = 0;
    uint32_t tile_height = 0;
    uint32_t tile_bpp = 0;
    uint32_t tile_stride = 0;

    for (const auto image : images) {
      if (image) {
        tile_width = image->get_width();
        tile_height = image->get_height();
        tile_bpp = image->get_bpp();
        tile_stride = image->get_stride();
        break;
      }
    }

    const auto output_stride = tile_width * num_cols * tile_bpp;
    output = image(tile_width * num_cols, tile_height * num_rows, tile_bpp, output_stride);

    for (std::size_t tile_y = 0; tile_y < num_rows; tile_y++) {
      for (std::size_t tile_x = 0; tile_x < num_cols; tile_x++) {
        const auto image_idx = tile_y * num_cols + tile_x;
        if (image_idx >= images.size()) {
          continue;
        }

        const auto image = images[image_idx];
        if (image == nullptr) {
          continue;
        }
        for (std::size_t y = 0; y < tile_height; y++) {
          std::copy_n(image->get_data() + y * tile_stride, tile_width * tile_bpp,
                      output.get_data() + output_stride * (tile_y * tile_height + y) +
                          tile_x * tile_width * tile_bpp);
        }
      }
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      std::vector<const image *> images;
      std::shared_ptr<stream_profile> profile;
      double timestamp{0.0};
      std::uint64_t frame_number{0};
      for (const auto &[name, field] : obj_msg->get_fields()) {
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field)) {
          names.insert(name);
        }
      }
      std::vector<std::string> name_list;
      std::copy(names.begin(), names.end(), std::back_inserter(name_list));
      std::sort(name_list.begin(), name_list.end());
      for (const auto &name : name_list) {
        if (const auto iter = obj_msg->get_fields().find(name);
            iter != obj_msg->get_fields().end()) {
          const auto &field = iter->second;
          if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field)) {
            const auto &img = image_msg->get_data();
            images.push_back(&img);
            profile = image_msg->get_profile();
            timestamp = image_msg->get_timestamp();
            frame_number = image_msg->get_frame_number();
          } else {
            images.push_back(nullptr);
          }
        } else {
          images.push_back(nullptr);
        }
      }

      image img;
      tiling(images, img);

      auto msg = std::make_shared<frame_message<image>>();

      msg->set_data(std::move(img));
      msg->set_profile(profile);
      msg->set_timestamp(timestamp);
      msg->set_frame_number(frame_number);

      output->send(msg);
    }
  }
};

class timestamp_node : public graph_node {
  graph_edge_ptr output;

 public:
  timestamp_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "timestamp_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {}

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto frame_msg = std::dynamic_pointer_cast<image_frame_message>(message)) {
      auto msg = std::make_shared<number_message>();
      msg->set_value(frame_msg->get_timestamp());
      output->send(msg);
    }
  }
};

class frame_demux_node : public graph_node {
 public:
  frame_demux_node() : graph_node() {}

  graph_edge_ptr add_output(std::string name) {
    auto outputs = get_outputs();
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      auto output = std::make_shared<graph_edge>(this);
      set_output(output, name);
      return output;
    }
    return it->second;
  }

  virtual std::string get_proc_name() const override { return "frame_demux_node"; }

  template <typename Archive>
  void save(Archive &archive) const {
    std::vector<std::string> output_names;
    auto outputs = get_outputs();
    for (auto output : outputs) {
      output_names.push_back(output.first);
    }
    archive(output_names);
  }

  template <typename Archive>
  void load(Archive &archive) {
    std::vector<std::string> output_names;
    archive(output_names);
    for (auto output_name : output_names) {
      set_output(std::make_shared<graph_edge>(this), output_name);
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto &obj_msg = frame_msg->get_data();
      for (auto field : obj_msg.get_fields()) {
        try {
          get_output(field.first)->send(field.second);
        } catch (const std::invalid_argument &e) {
          spdlog::warn(e.what());
        }
      }
    }
  }
};

class frame_number_numbering_node : public graph_node {
  uint64_t frame_number;
  graph_edge_ptr output;

 public:
  frame_number_numbering_node()
      : graph_node(), frame_number(0), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "frame_number_numbering_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(frame_number);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      msg->set_frame_number(frame_number++);
      output->send(msg);
    }
    if (auto msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto &[name, field] : msg->get_fields()) {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(field)) {
          frame_msg->set_frame_number(frame_number);
        }
      }
      frame_number++;
      output->send(msg);
    }
  }
};

template <typename Task>
class task_queue {
  const uint32_t thread_count;
  std::unique_ptr<std::thread[]> threads;
  std::queue<Task> tasks{};
  std::mutex tasks_mutex;
  std::condition_variable condition;
  std::atomic_bool running{true};

  void worker() {
    for (;;) {
      Task task;

      {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        condition.wait(lock, [&] { return !tasks.empty() || !running; });
        if (!running) {
          return;
        }

        task = std::move(tasks.front());
        tasks.pop();
      }

      task();
    }
  }

 public:
  task_queue(const uint32_t thread_count = std::thread::hardware_concurrency())
      : thread_count(thread_count), threads(std::make_unique<std::thread[]>(thread_count)) {
    for (uint32_t i = 0; i < thread_count; ++i) {
      threads[i] = std::thread(&task_queue::worker, this);
    }
  }

  void push_task(const Task &task) {
    {
      const std::lock_guard<std::mutex> lock(tasks_mutex);

      if (!running) {
        throw std::runtime_error("Cannot schedule new task after shutdown.");
      }

      tasks.push(task);
    }

    condition.notify_one();
  }
  ~task_queue() {
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      running = false;
    }

    condition.notify_all();

    for (uint32_t i = 0; i < thread_count; ++i) {
      threads[i].join();
    }
  }
  size_t size() const { return tasks.size(); }
};

class parallel_queue_node : public graph_node {
  std::unique_ptr<task_queue<std::function<void()>>> workers;
  graph_edge_ptr output;

  uint32_t num_threads;

 public:
  parallel_queue_node()
      : graph_node(),
        workers(),
        output(std::make_shared<graph_edge>(this)),
        num_threads(std::thread::hardware_concurrency()) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "parallel_queue_node"; }

  void set_num_threads(uint32_t value) { num_threads = value; }
  uint32_t get_num_threads() const { return num_threads; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(num_threads);
  }

  virtual void run() override { workers.reset(new task_queue<std::function<void()>>(num_threads)); }

  virtual void stop() override { workers.reset(); }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      workers->push_task([this, msg]() { output->send(msg); });
    }
  }
};

class greater_graph_message_ptr {
 public:
  bool operator()(const graph_message_ptr &lhs, const graph_message_ptr &rhs) const {
    return std::dynamic_pointer_cast<frame_message_base>(lhs)->get_frame_number() >
           std::dynamic_pointer_cast<frame_message_base>(rhs)->get_frame_number();
  }
};

class frame_number_ordering_node : public graph_node {
  graph_edge_ptr output;
  std::mutex mtx;

  std::priority_queue<graph_message_ptr, std::deque<graph_message_ptr>, greater_graph_message_ptr>
      messages;

  std::shared_ptr<std::thread> th;
  std::atomic_bool running;
  std::condition_variable cv;
  std::uint32_t max_size;
  std::atomic_ullong frame_number;

 public:
  frame_number_ordering_node()
      : graph_node(), output(std::make_shared<graph_edge>(this)), max_size(100), frame_number(0) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "frame_number_ordering_node"; }

  void set_max_size(std::uint32_t value) { max_size = value; }
  std::uint32_t get_max_size() const { return max_size; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(max_size, frame_number);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (!running) {
      return;
    }

    if (input_name == "default") {
      std::lock_guard<std::mutex> lock(mtx);

      if (messages.size() >= max_size) {
        std::cout << "Fifo overflow" << std::endl;
        spdlog::error("Fifo overflow");
      } else {
        messages.push(message);
        cv.notify_one();
      }
    }
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([this]() {
      while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] {
          return (!messages.empty() && std::dynamic_pointer_cast<frame_message_base>(messages.top())
                                               ->get_frame_number() == frame_number) ||
                 !running;
        });

        if (!running) {
          break;
        }
        if (!messages.empty() &&
            std::dynamic_pointer_cast<frame_message_base>(messages.top())->get_frame_number() ==
                frame_number) {
          const auto message = messages.top();
          messages.pop();
          output->send(message);

          frame_number++;
        }
      }
    }));
  }

  virtual void stop() override {
    if (running.load()) {
      {
        std::lock_guard<std::mutex> lock(mtx);
        running.store(false);
      }
      cv.notify_one();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }
};
}  // namespace coalsack

COALSACK_REGISTER_MESSAGE(coalsack::image_message, coalsack::graph_message)

COALSACK_REGISTER_MESSAGE(coalsack::frame_message_base, coalsack::graph_message)

COALSACK_REGISTER_MESSAGE(coalsack::blob_frame_message, coalsack::frame_message_base)

COALSACK_REGISTER_MESSAGE(coalsack::image_frame_message, coalsack::frame_message_base)

COALSACK_REGISTER_NODE(coalsack::image_heartbeat_node, coalsack::heartbeat_node)

COALSACK_REGISTER_NODE(coalsack::approximate_time_sync_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_number_sync_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::tiling_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::timestamp_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_demux_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_number_numbering_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::parallel_queue_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_number_ordering_node, coalsack::graph_node)
