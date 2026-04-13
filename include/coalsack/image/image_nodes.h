/// @file image_nodes.h
/// @brief Image processing nodes: synchronization, tiling, ordering, and more.
/// @ingroup image
#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cereal/types/atomic.hpp>
#include <cereal/types/base_class.hpp>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "coalsack/image/frame_message.h"
#include "coalsack/image/image_message.h"
#include "coalsack/messages/number_message.h"
#include "coalsack/messages/object_message.h"
#include "coalsack/nodes/heartbeat_node.h"
#include "coalsack/util/syncer.h"

namespace coalsack {

/// @brief Periodic source node that emits a static image as a @c frame_message at a fixed rate.
/// @details On each heartbeat tick, wraps the stored @c image in a @c frame_message<image>
///          stamped with the current system time and sends it on @b "default".
/// @par Inputs
///   (none — autonomous source)
/// @par Outputs
/// - @b "default" — @c frame_message<image>
/// @par Properties
/// - interval (uint32_t, inherited) — emission period in milliseconds (see heartbeat_node)
/// @see heartbeat_node, video_capture_node
/// @ingroup image
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

/// @brief Multi-stream synchronizer node that collects frames from named inputs and emits
///        an @c object_message when all streams have a temporally-aligned set of frames.
/// @details Collects one @c frame_message from each registered input port.  When all ports have
///          a frame whose sync key (timestamp or frame number, depending on the Config policy)
///          is within the tolerance window, emits an @c object_message bundling them.
/// @ingroup image
/// @tparam Config Synchronization policy; defaults to @c approximate_time_sync_config.
///         Use @c frame_number_sync_config (aliased as @c frame_number_sync_node) for
///         frame-number-based synchronization.
/// @par Inputs
/// - @b "{stream_name}" — @c frame_message_base (one input port per registered stream)
/// @par Outputs
/// - @b "default" — @c object_message with one @c frame_message per stream key
/// @par Properties
/// - interval (double, 0.0) — synchronization tolerance window in milliseconds
/// @see tiling_node, frame_demux_node
template <typename Config = approximate_time_sync_config>
class sync_node : public graph_node {
  using sync_config = Config;
  using sync_info = typename sync_config::sync_info;
  using syncer_type = stream_syncer<graph_message_ptr, std::string, sync_info>;
  using stream_id_type = std::string;

  syncer_type syncer;
  graph_edge_ptr output;
  sync_config config;
  std::vector<stream_id_type> initial_ids;

 public:
  sync_node()
      : graph_node(),
        syncer(),
        output(std::make_shared<graph_edge>(this)),
        config(),
        initial_ids() {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "sync_node"; }

  void set_config(sync_config config) { this->config = config; }
  const sync_config &get_config() const { return config; }
  sync_config &get_config() { return config; }

  void set_initial_ids(const std::vector<stream_id_type> &ids) { this->initial_ids = ids; }
  const std::vector<stream_id_type> &get_initial_ids() const { return initial_ids; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(config, initial_ids);
  }

  virtual void run() override {
    syncer.start(std::make_shared<typename syncer_type::callback_type>(
                     [this](const std::map<std::string, graph_message_ptr> &frames) {
                       auto msg = std::make_shared<object_message>();
                       for (auto frame : frames) {
                         msg->add_field(frame.first, frame.second);
                       }
                       output->send(msg);
                     }),
                 initial_ids);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      syncer.sync(input_name, frame_msg, config.create_sync_info(frame_msg));
    }
  }
};

using approximate_time_sync_node = sync_node<approximate_time_sync_config>;
using frame_number_sync_node = sync_node<frame_number_sync_config>;

/// @brief Tiles multiple input @c frame_message<image> frames into a single composite image.
/// @details Accumulates one @c frame_message<image> per registered column name.  When all
///          columns have a new frame, composites them side-by-side (column-major) into a single
///          wide image and emits it on @b "default".
/// @par Inputs
/// - @b "{stream_name}" — @c frame_message<image> (one per column, registered via @c add_image())
/// @par Outputs
/// - @b "default" — @c frame_message<image> containing the tiled composite
/// @par Properties
/// - num_cols (uint32_t) — number of tile columns in the grid
/// - num_rows (uint32_t) — number of tile rows in the grid
/// @see sync_node, frame_demux_node
/// @ingroup image
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

/// @brief Extracts the timestamp from an incoming @c image_frame_message and emits it as a @c number_message.
/// @details Casts each incoming message to @c image_frame_message; on success emits a
///          @c number_message containing @c get_timestamp() in microseconds.
/// @par Inputs
/// - @b "default" — @c image_frame_message (non-matching types are discarded)
/// @par Outputs
/// - @b "default" — @c number_message containing the frame timestamp in microseconds
/// @par Properties
///   (none)
/// @see frame_message, frame_number_numbering_node
/// @ingroup image
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

/// @brief Demultiplexes a @c frame_message<object_message> into its constituent labeled streams.
/// @details Extracts per-key child messages from an incoming @c frame_message<object_message>
///          and republishes each child on the named output port that matches its field key.
/// @par Inputs
/// - @b "default" — @c frame_message<object_message>
/// @par Outputs
/// - @b "<field_key>" — child @c graph_message for each field in the @c object_message
/// @par Properties
///   (none — output ports are registered at build time via add_output())
/// @see sync_node, tiling_node
/// @ingroup image
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

/// @brief Assigns monotonically increasing frame numbers to incoming @c frame_message_base messages.
/// @details Maintains an internal counter incremented for each arriving message.  If the input is
///          an @c object_message, the counter is applied to each contained @c frame_message_base field.
/// @par Inputs
/// - @b "default" — @c frame_message_base or @c object_message wrapping @c frame_message_base fields
/// @par Outputs
/// - @b "default" — the same message with frame_number set
/// @par Properties
///   (none)
/// @see frame_number_ordering_node, timestamp_node
/// @ingroup image
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

/// @brief Forwards frame messages through a fixed-size thread pool, enabling parallel processing.
/// @details Distributes incoming @c frame_message_base messages across worker threads; each worker
///          emits its result on @b "default" when complete.  Output order is not guaranteed.
/// @par Inputs
/// - @b "default" — @c frame_message_base
/// @par Outputs
/// - @b "default" — @c frame_message_base (forwarded asynchronously from worker threads)
/// @par Properties
/// - num_threads (uint32_t, default hardware_concurrency) — worker thread count
/// @see frame_number_ordering_node, fifo_node
/// @ingroup image
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

  virtual void initialize() override { workers.reset(new task_queue<std::function<void()>>(num_threads)); }

  virtual void finalize() override { workers.reset(); }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      if (workers) {
        workers->push_task([this, msg]() { output->send(msg); });
      }
    }
  }
};

/// @brief Comparator for a min-heap of frame_message_base by frame number (ascending order).
class greater_graph_message_ptr {
 public:
  bool operator()(const graph_message_ptr &lhs, const graph_message_ptr &rhs) const {
    return std::dynamic_pointer_cast<frame_message_base>(lhs)->get_frame_number() >
           std::dynamic_pointer_cast<frame_message_base>(rhs)->get_frame_number();
  }
};

/// @brief Reorders out-of-order @c frame_message_base messages by frame number using a bounded priority queue.
/// @details Buffers incoming frames in a min-heap keyed by frame number.  When the expected next
///          frame number arrives, or the buffer reaches @c max_size, messages are flushed in order.
/// @par Inputs
/// - @b "default" — @c frame_message_base (may arrive out of order)
/// @par Outputs
/// - @b "default" — @c frame_message_base in frame-number order
/// @par Properties
/// - max_size (uint32_t, default 10) — maximum number of buffered frames before oldest is forwarded
/// @see parallel_queue_node, frame_number_numbering_node
/// @ingroup image
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

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::image_heartbeat_node, coalsack::heartbeat_node)

COALSACK_REGISTER_NODE(coalsack::approximate_time_sync_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_number_sync_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::tiling_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::timestamp_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_demux_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_number_numbering_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::parallel_queue_node, coalsack::graph_node)

COALSACK_REGISTER_NODE(coalsack::frame_number_ordering_node, coalsack::graph_node)
