find_package(realsense2)
find_package(PkgConfig)
if(${PkgConfig_FOUND})
    pkg_check_modules(LIBCAMERA libcamera)
endif()

set(ENABLE_RS_D435_EXT ${realsense2_FOUND})
set(ENABLE_LIBCAMERA_EXT ${LIBCAMERA_FOUND})

message("ENABLE_LIBCAMERA_EXT: ${ENABLE_LIBCAMERA_EXT}")
message("ENABLE_RS_D435_EXT: ${ENABLE_RS_D435_EXT}")

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

include_directories(
    SYSTEM ${CMAKE_BINARY_DIR}/libjpeg-turbo/include
    SYSTEM ${CMAKE_BINARY_DIR}/lz4/include
)

include(${CMAKE_CURRENT_LIST_DIR}/server/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/greeting/CMakeLists.txt)
