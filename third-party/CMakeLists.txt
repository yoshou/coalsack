string(REPLACE ${PROJECT_SOURCE_DIR}/ "" _rel_path ${CMAKE_CURRENT_LIST_DIR})

include(ExternalProject)

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

ExternalProject_Add (cereal
    PREFIX cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/cereal"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Add (spdlog
    PREFIX spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG "v1.8.2"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/spdlog"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.10.0
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/googletest"
    BINARY_DIR "${CMAKE_BINARY_DIR}/third-party/googletest-build"
    INSTALL_COMMAND ""
)

include_directories(
    ${CMAKE_BINARY_DIR}/third-party/cereal/include
    ${CMAKE_BINARY_DIR}/third-party/spdlog/include
    ${CMAKE_BINARY_DIR}/third-party/googletest/googletest/include
)
