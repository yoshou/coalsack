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
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:realsense2::realsense2>
)

target_compile_definitions(greeting PUBLIC
    FMT_HEADER_ONLY
)
