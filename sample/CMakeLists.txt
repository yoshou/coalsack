find_package(realsense2)
find_package(PkgConfig)
if(${PkgConfig_FOUND})
    pkg_check_modules(LIBCAMERA libcamera)
endif()

set(ENABLE_RS_D435_EXT ${realsense2_FOUND})
set(ENABLE_LIBCAMERA_EXT ${LIBCAMERA_FOUND})
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ENABLE_ONNXRUNTIME_EXT 1)
else()
    set(ENABLE_ONNXRUNTIME_EXT 0)
endif()
if(DEFINED ENV{TENSORFLOW_ROOT})
    set(ENABLE_TFLITE_EXT 1)
else()
    set(ENABLE_TFLITE_EXT 0)
endif()

message("ENABLE_LIBCAMERA_EXT: ${ENABLE_LIBCAMERA_EXT}")
message("ENABLE_RS_D435_EXT: ${ENABLE_RS_D435_EXT}")
message("ENABLE_ONNXRUNTIME_EXT: ${ENABLE_ONNXRUNTIME_EXT}")
message("ENABLE_TFLITE_EXT: ${ENABLE_TFLITE_EXT}")

if(${ENABLE_ONNXRUNTIME_EXT})
    set(ONNXRUNTIME_INCLUDE_DIRS $ENV{ONNXRUNTIME_ROOT}/include)
    set(ONNXRUNTIME_LIBS $ENV{ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

if(${ENABLE_TFLITE_EXT})
    add_subdirectory(
        "$ENV{TENSORFLOW_ROOT}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
        EXCLUDE_FROM_ALL
    )
endif()

ExternalProject_Add (libjpeg-turbo
    PREFIX libjpeg-turbo
    GIT_REPOSITORY "https://github.com/libjpeg-turbo/libjpeg-turbo.git"
    GIT_TAG "2.1.4"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/libjpeg-turbo"
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/libjpeg-turbo"
        "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
)

ExternalProject_Add (lz4
    PREFIX lz4
    GIT_REPOSITORY "https://github.com/lz4/lz4.git"
    GIT_TAG "dev"
    SOURCE_SUBDIR "build/cmake"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/lz4"
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/lz4"
        "-DBUILD_STATIC_LIBS=ON"
)

ExternalProject_Add (fmt
    PREFIX fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG "10.0.0"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/fmt"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Add(json
    PREFIX json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.10.2
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/json"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

include_directories(
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/libjpeg-turbo/include
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/lz4/include
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/fmt/include
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/json/include
)

include(${CMAKE_CURRENT_LIST_DIR}/server/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/greeting/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/dnn/CMakeLists.txt)
