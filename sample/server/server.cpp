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
#include "graph_proc_img.h"
#include "graph_proc_cv.h"

#ifdef ENABLE_RS_D435_EXT
#include "ext/graph_proc_rs_d435.h"
#endif
#ifdef ENABLE_LIBCAMERA_EXT
#include "ext/graph_proc_libcamera.h"
#endif
#ifdef ENABLE_DEPTHAI_EXT
#include "ext/graph_proc_depthai.h"
#endif
#include "ext/graph_proc_jpeg.h"
#include "ext/graph_proc_lz4.h"
#include "ext/graph_proc_cv_ext.h"

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
    
    asio::io_context io_context;

    graph_proc_server server(io_context, "0.0.0.0", 31400);

    on_shutdown_handlers.push_back([&io_context] {
        io_context.stop();
    });

    io_context.run();

    return 0;
}
catch (std::exception &e)
{
    std::cout << e.what() << std::endl;
    shutdown();
}
