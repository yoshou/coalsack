add_executable(superpoint_lightglue
    ${CMAKE_CURRENT_LIST_DIR}/superpoint_lightglue.cpp
)

add_dependencies(superpoint_lightglue
    cereal
    spdlog
    libjpeg-turbo
    json
)

target_include_directories(superpoint_lightglue PUBLIC
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:${LIBCAMERA_INCLUDE_DIRS}>
)

target_link_libraries(superpoint_lightglue
    ${OpenCV_LIBS}
    coalsack
    pthread
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:${LIBCAMERA_LIBS}>
)

install(TARGETS superpoint_lightglue DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
