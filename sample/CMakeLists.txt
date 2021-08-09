cmake_minimum_required(VERSION 3.8)

project(graph_proc_server)

include_directories(
    SYSTEM ${CMAKE_BINARY_DIR}/libjpeg-turbo/include
)
add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/server.cpp
)

add_dependencies(${PROJECT_NAME}
    cereal
    spdlog
    libjpeg-turbo
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    coalsack
    pthread
    boost_system
    boost_thread
    realsense2
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
)

project(graph_proc_client)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/client.cpp
)

add_dependencies(${PROJECT_NAME}
    cereal
    spdlog
    libjpeg-turbo
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    coalsack
    pthread
    stdc++fs
    boost_system
    boost_thread
    boost_program_options
    realsense2
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
)
