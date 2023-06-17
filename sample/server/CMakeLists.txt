add_executable(graph_proc_server
    ${CMAKE_CURRENT_LIST_DIR}/server.cpp
)

add_dependencies(graph_proc_server
    cereal
    spdlog
    libjpeg-turbo
)

target_link_libraries(graph_proc_server
    ${OpenCV_LIBS}
    coalsack
    pthread
    realsense2::realsense2
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
)