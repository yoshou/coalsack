#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio.hpp>

// Core
#include "coalsack/core/graph_proc_server.h"
#include "coalsack/core/subgraph.h"
// Messages
#include "coalsack/messages/blob_message.h"
#include "coalsack/messages/list_message.h"
#include "coalsack/messages/number_message.h"
#include "coalsack/messages/object_message.h"
#include "coalsack/messages/text_message.h"
// Nodes
#include "coalsack/nodes/buffer_node.h"
#include "coalsack/nodes/callback_nodes.h"
#include "coalsack/nodes/clock_buffer_node.h"
#include "coalsack/nodes/console_node.h"
#include "coalsack/nodes/demux_node.h"
#include "coalsack/nodes/fifo_node.h"
#include "coalsack/nodes/heartbeat_node.h"
#include "coalsack/nodes/mux_node.h"
#include "coalsack/nodes/passthrough_node.h"
// Network
#include "coalsack/network/broadcast_listener_node.h"
#include "coalsack/network/broadcast_talker_node.h"
#include "coalsack/network/p2p_listener_node.h"
#include "coalsack/network/p2p_talker_node.h"
#include "coalsack/network/p2p_tcp_listener_node.h"
#include "coalsack/network/p2p_tcp_talker_node.h"
// Image
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/image/image_nodes.h"

#ifdef ENABLE_RS_D435_EXT
#include "coalsack/ext/graph_proc_rs_d435.h"
#endif
#ifdef ENABLE_LIBCAMERA_EXT
#include "coalsack/ext/graph_proc_libcamera.h"
#endif
#ifdef ENABLE_DEPTHAI_EXT
#include "coalsack/ext/graph_proc_depthai.h"
#endif
#include "coalsack/ext/graph_proc_cv_ext.h"
#include "coalsack/ext/graph_proc_jpeg.h"
#include "coalsack/ext/graph_proc_lz4.h"

using namespace coalsack;

static std::vector<std::function<void()>> on_shutdown_handlers;

static void shutdown() {
  for (auto handler : on_shutdown_handlers) {
    handler();
  }
}

static void sigint_handler(int) {
  shutdown();
  exit(0);
}

int main(int argc, char *argv[]) try {
  signal(SIGINT, sigint_handler);

  spdlog::set_level(spdlog::level::debug);

  asio::io_context io_context;

  graph_proc_server server(io_context, "0.0.0.0", 31400);

  on_shutdown_handlers.push_back([&io_context] { io_context.stop(); });

  io_context.run();

  return 0;
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  shutdown();
}
