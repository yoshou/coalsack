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

#include "coalsack/core/graph_proc_client.h"
#include "coalsack/core/subgraph.h"
#include "coalsack/network/p2p_listener_node.h"
#include "coalsack/network/p2p_talker_node.h"
#include "coalsack/nodes/buffer_node.h"
#include "coalsack/nodes/console_node.h"
#include "coalsack/nodes/heartbeat_node.h"
#include "coalsack/nodes/passthrough_node.h"

using namespace coalsack;

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic<bool> quit(false);

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

  asio::io_context io_context;

  std::shared_ptr<subgraph> g1(new subgraph());
  std::shared_ptr<subgraph> g2(new subgraph());

  std::shared_ptr<text_heartbeat_node> n1(new text_heartbeat_node());
  n1->set_message("Hello!!\n");
  g1->add_node(n1);

  std::shared_ptr<p2p_talker_node> n2(new p2p_talker_node());
  n2->set_input(n1->get_output());
  g1->add_node(n2);

  std::shared_ptr<p2p_listener_node> n3(new p2p_listener_node());
  n3->set_input(n2->get_output());
  n3->set_endpoint("127.0.0.1", 0);
  g2->add_node(n3);

  std::shared_ptr<passthrough_node> n4(new passthrough_node());
  n4->set_input(n3->get_output());
  g2->add_node(n4);

  std::shared_ptr<buffer_node> n5(new buffer_node());
  n5->set_input(n4->get_output());
  n5->set_interval(2000);
  g2->add_node(n5);

  std::shared_ptr<console_node> n6(new console_node());
  n6->set_input(n5->get_output());
  g2->add_node(n6);

  graph_proc_client client;
  client.deploy(io_context, "127.0.0.1", 31400, g1);
  client.deploy(io_context, "127.0.0.1", 31400, g2);

  client.run();

  on_shutdown_handlers.push_back([&io_context] { io_context.stop(); });

  io_context.run();

  std::this_thread::sleep_for(std::chrono::milliseconds(10000));

  return 0;
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  shutdown();
  return 0;
}
