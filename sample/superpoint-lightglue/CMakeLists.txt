# Restore ONNX files from chunks before building
set(DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
set(ONNX_FILES
    ${DATA_DIR}/superpoint_lightglue_pipeline.onnx
)

# Check if ONNX files need to be restored
set(NEED_RESTORE FALSE)
foreach(ONNX_FILE ${ONNX_FILES})
    if(NOT EXISTS ${ONNX_FILE})
        set(NEED_RESTORE TRUE)
        break()
    endif()
endforeach()

if(NEED_RESTORE)
    message(STATUS "Restoring ONNX files from chunks...")
    execute_process(
        COMMAND python3 ${DATA_DIR}/restore_onnx.py --all ${DATA_DIR}
        WORKING_DIRECTORY ${DATA_DIR}
        RESULT_VARIABLE RESTORE_RESULT
    )
    if(NOT RESTORE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to restore ONNX files")
    endif()
endif()

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
