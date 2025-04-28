#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <boost/asio.hpp>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "ext/graph_proc_depthai.h"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"

using namespace coalsack;

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown() {
  for (auto handler : on_shutdown_handlers) {
    handler();
  }
  exit_flag.store(true);
}

static void sigint_handler(int) {
  shutdown();
  exit(0);
}

class local_server {
  asio::io_context io_context;
  std::shared_ptr<graph_proc_server> server;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  local_server()
      : io_context(),
        server(std::make_shared<graph_proc_server>(io_context, "0.0.0.0", 31400)),
        th(),
        running(false) {}

  void run() {
    running = true;
    th.reset(new std::thread([this] { io_context.run(); }));
  }

  void stop() {
    if (running.load()) {
      running.store(false);
      io_context.stop();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }

  ~local_server() { stop(); }
};

int main(int argc, char *argv[]) try {
  signal(SIGINT, sigint_handler);

  spdlog::set_level(spdlog::level::debug);

  local_server server;
  server.run();

  asio::io_context io_context;

  std::shared_ptr<subgraph> g(new subgraph());

  std::shared_ptr<coalsack::depthai_color_camera_node> capture(
      new coalsack::depthai_color_camera_node());
  g->add_node(capture);

  std::shared_ptr<video_viz_node> viz(new video_viz_node());
  viz->set_input(capture->get_output());
  viz->set_image_name("video");
  g->add_node(viz);

  graph_proc_client client;
  client.deploy(io_context, "127.0.0.1", 31400, g);

  on_shutdown_handlers.push_back([&client, &server] {
    client.stop();
    server.stop();
  });

  std::thread io_thread([&io_context] { io_context.run(); });

  client.run();

  while (!exit_flag.load()) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<std::int64_t>(1000.0 / 30.0)));
    if (viz) {
      cv_window::wait_key(1);
    }
  }

  if (io_thread.joinable()) {
    io_thread.join();
  }

  return 0;
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  shutdown();
}