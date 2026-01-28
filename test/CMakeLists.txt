add_executable(tests
    ${CMAKE_CURRENT_LIST_DIR}/syncer_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tensor_test.cpp
)

add_dependencies(tests
    googletest
    spdlog
)

include_directories(
    SYSTEM ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(tests
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest.a
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest_main.a
    pthread
)

target_compile_definitions(tests PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_superpoint_lightglue
    ${CMAKE_CURRENT_LIST_DIR}/test_superpoint_lightglue.cpp
)

add_dependencies(test_superpoint_lightglue
    spdlog
)

target_include_directories(test_superpoint_lightglue PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_superpoint_lightglue
    coalsack
    pthread
    protobuf::libprotobuf
    ONNX::onnx
    ONNX::onnx_proto
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)

target_compile_definitions(test_superpoint_lightglue PUBLIC
    FMT_HEADER_ONLY
    ENABLE_ONNX_IMPORT
)

add_executable(test_gguf_loader
    ${CMAKE_CURRENT_LIST_DIR}/test_gguf_loader.cpp
)

add_dependencies(test_gguf_loader
    spdlog
)

target_include_directories(test_gguf_loader PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_gguf_loader
    coalsack
    pthread
)

target_compile_definitions(test_gguf_loader PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_tokenizer
    ${CMAKE_CURRENT_LIST_DIR}/test_tokenizer.cpp
)

add_dependencies(test_tokenizer
    spdlog
)

target_include_directories(test_tokenizer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_tokenizer
    coalsack
    pthread
)

target_compile_definitions(test_tokenizer PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_chat_template
    ${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp
)

add_dependencies(test_chat_template
    spdlog
)

target_include_directories(test_chat_template PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_chat_template
    coalsack
    nlohmann_json::nlohmann_json
    pthread
)

target_compile_definitions(test_chat_template PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_frame_sync
    ${CMAKE_CURRENT_LIST_DIR}/test_frame_sync.cpp
)

add_dependencies(test_frame_sync
    googletest
    spdlog
)

target_include_directories(test_frame_sync PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_frame_sync
    coalsack
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest.a
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest_main.a
    pthread
)

target_compile_definitions(test_frame_sync PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_nn_nodes
    ${CMAKE_CURRENT_LIST_DIR}/test_nn_nodes.cpp
)

add_dependencies(test_nn_nodes
    googletest
    spdlog
)

target_include_directories(test_nn_nodes PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_nn_nodes
    coalsack
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest.a
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest_main.a
    pthread
)

target_compile_definitions(test_nn_nodes PUBLIC
    FMT_HEADER_ONLY
)
