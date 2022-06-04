#include <chrono>
#include <string>
#include <atomic>
#include <thread>
#include <functional>
#include <vector>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <boost/asio.hpp>

#include <signal.h>
#include <unistd.h>

#include "graph_proc.h"

using namespace coalsack;

static std::vector<std::function<void()>> on_shutdown_handlers;

static void shutdown()
{
    for (auto handler : on_shutdown_handlers)
    {
        handler();
    }
}

static void sigint_handler(int)
{
    shutdown();
    exit(0);
}

int main(int argc, char *argv[])
try
{
    signal(SIGINT, sigint_handler);

    spdlog::set_level(spdlog::level::debug);
    
    asio::io_service io_service;

    graph_proc_server server(io_service, "0.0.0.0", 31400);

    on_shutdown_handlers.push_back([&io_service] {
        io_service.stop();
    });

    io_service.run();

    return 0;
}
catch (std::exception &e)
{
    std::cout << e.what() << std::endl;
    shutdown();
}
