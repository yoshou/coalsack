add_executable(graph_proc_server
    ${CMAKE_CURRENT_LIST_DIR}/server.cpp
)

add_dependencies(graph_proc_server
    cereal
    spdlog
    libjpeg-turbo
)

target_include_directories(graph_proc_server PUBLIC
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(graph_proc_server
    ${OpenCV_LIBS}
    coalsack
    pthread
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
    ${CMAKE_BINARY_DIR}/lz4/lib/liblz4.a
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:realsense2::realsense2>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:camera>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:camera-base>
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:ENABLE_RS_D435_EXT>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:ENABLE_LIBCAMERA_EXT>
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:ENABLE_ONNXRUNTIME_EXT>
)