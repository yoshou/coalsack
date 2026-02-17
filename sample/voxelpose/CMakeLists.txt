# Restore ONNX files from chunks before building
set(DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
set(ONNX_FILES
    ${DATA_DIR}/backbone.onnx
    ${DATA_DIR}/pose_v2v_net.onnx
    ${DATA_DIR}/proposal_v2v_net.onnx
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

add_executable(voxelpose
    ${CMAKE_CURRENT_LIST_DIR}/voxelpose.cpp
)

add_dependencies(voxelpose
    cereal
    spdlog
    libjpeg-turbo
    json
)

target_include_directories(voxelpose PUBLIC
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:${LIBCAMERA_INCLUDE_DIRS}>
)

target_link_libraries(voxelpose
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
    FMT_HEADER_ONLY
    $<$<BOOL:${ENABLE_RS_D435_EXT}>:ENABLE_RS_D435_EXT>
    $<$<BOOL:${ENABLE_LIBCAMERA_EXT}>:ENABLE_LIBCAMERA_EXT>
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:ENABLE_ONNXRUNTIME_EXT>
)