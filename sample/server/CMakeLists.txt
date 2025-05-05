add_executable(graph_proc_server
    ${CMAKE_CURRENT_LIST_DIR}/server.cpp
)

add_dependencies(graph_proc_server
    cereal
    spdlog
    libjpeg-turbo
    lz4
)

target_include_directories(graph_proc_server PUBLIC
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:${LIBCAMERA_INCLUDE_DIRS}>
)

if(${ENABLE_LIBCAMERA_EXT})
    set(LIBCAMERA_LIBRARIES ${LIBCAMERA_LINK_LIBRARIES})
endif()

target_link_libraries(graph_proc_server
    ${OpenCV_LIBS}
    coalsack
    pthread
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
    ${CMAKE_BINARY_DIR}/lz4/lib/liblz4.a
    $<$<BOOL:${ENABLE_TFLITE_EXT}>:depthai::core>
    $<$<BOOL:${ENABLE_TFLITE_EXT}>:depthai::opencv>
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:realsense2::realsense2>
    ${LIBCAMERA_LIBRARIES}
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    $<$<BOOL:${ENABLE_TFLITE_EXT}>:tensorflow-lite>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    FMT_HEADER_ONLY
    $<$<BOOL:${ENABLE_DEPTHAI_EXT}>:ENABLE_DEPTHAI_EXT>
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:ENABLE_RS_D435_EXT>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:ENABLE_LIBCAMERA_EXT>
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:ENABLE_ONNXRUNTIME_EXT>
    $<$<BOOL:${ENABLE_TFLITE_EXT}>:ENABLE_TFLITE_EXT>
)