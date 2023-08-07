add_executable(faster_voxelpose
    ${CMAKE_CURRENT_LIST_DIR}/faster_voxelpose.cpp
)

add_dependencies(faster_voxelpose
    cereal
    spdlog
    libjpeg-turbo
    json
)

target_include_directories(faster_voxelpose PUBLIC
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:${LIBCAMERA_INCLUDE_DIRS}>
)

target_link_libraries(faster_voxelpose
    ${OpenCV_LIBS}
    coalsack
    pthread
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
    ${CMAKE_BINARY_DIR}/lz4/lib/liblz4.a
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:realsense2::realsense2>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:camera>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:camera-base>
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    $<$<BOOL:${ENABLE_TFLITE_EXT}>:tensorflow-lite>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:ENABLE_RS_D435_EXT>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:ENABLE_LIBCAMERA_EXT>
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:ENABLE_ONNXRUNTIME_EXT>
    $<$<BOOL:${ENABLE_TFLITE_EXT}>:ENABLE_TFLITE_EXT>
)