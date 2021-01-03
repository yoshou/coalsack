cmake_minimum_required(VERSION 3.8)

project(coalsack)

set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall -Wextra ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE "-O2 -Wall -Wextra ${CMAKE_C_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -Wextra ${CMAKE_CXX_FLAGS_RELEASE}")

add_library(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/src/rpc_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/rpc_client.cpp
)

target_sources(${PROJECT_NAME}
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include/rpc_client.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/rpc_server.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/rpc_common.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/data_stream_common.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/data_stream_receiver.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/data_stream_transmitter.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/graph_proc.h"
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

include(${CMAKE_CURRENT_LIST_DIR}/third-party/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sample/CMakeLists.txt)
