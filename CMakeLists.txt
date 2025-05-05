cmake_minimum_required(VERSION 3.8)

project(coalsack)

find_package(Iconv REQUIRED)
find_package(OpenCV 4.0 REQUIRED)
find_package(Boost REQUIRED)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_RELEASE "-O2")

    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
    endif()
endif()

add_library(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/src/rpc_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/rpc_client.cpp
)

add_dependencies(${PROJECT_NAME}
    cereal
    spdlog
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
    "${CMAKE_CURRENT_LIST_DIR}/include/graph_proc_img.h"
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${Boost_INCLUDE_DIRS}
)

include(${CMAKE_CURRENT_LIST_DIR}/third-party/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sample/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
