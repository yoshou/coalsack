cmake_minimum_required(VERSION 3.8)

project(coalsack)

find_package(OpenCV 4.0 REQUIRED)
find_package(Boost REQUIRED)
find_package(realsense2)
find_package(PkgConfig)
if(${PkgConfig_FOUND})
    pkg_check_modules(LIBCAMERA libcamera)
endif()

set(ENABLE_RS_D435_EXT ${realsense2_FOUND})
set(ENABLE_LIBCAMERA_EXT ${LIBCAMERA_FOUND})

message("ENABLE_LIBCAMERA_EXT: ${ENABLE_LIBCAMERA_EXT}")
message("ENABLE_RS_D435_EXT: ${ENABLE_RS_D435_EXT}")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
ENDIF()

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
    "${CMAKE_CURRENT_LIST_DIR}/include/graph_proc_img.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/ext/graph_proc_rs_d435.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/ext/graph_proc_jpeg.h"
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

include(${CMAKE_CURRENT_LIST_DIR}/third-party/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sample/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
