add_executable(greeting
    ${CMAKE_CURRENT_LIST_DIR}/greeting.cpp
)

add_dependencies(greeting
    cereal
    spdlog
    libjpeg-turbo
)

target_link_libraries(greeting
    ${OpenCV_LIBS}
    coalsack
    pthread
    stdc++fs
    realsense2::realsense2
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
)
