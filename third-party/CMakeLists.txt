string(REPLACE ${PROJECT_SOURCE_DIR}/ "" _rel_path ${CMAKE_CURRENT_LIST_DIR})

include(ExternalProject)

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

include_directories(
    ${CMAKE_BINARY_DIR}/third-party/cereal/include
    ${CMAKE_BINARY_DIR}/third-party/spdlog/include
)
