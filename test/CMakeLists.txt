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

add_executable(test_llama_backend
    ${CMAKE_CURRENT_LIST_DIR}/test_llama_backend.cpp
)

target_link_libraries(test_llama_backend
    coalsack
    llama
    pthread
)

add_executable(test_generation
    ${CMAKE_CURRENT_LIST_DIR}/test_generation.cpp
)

target_include_directories(test_generation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_generation
    coalsack
    llama
    nlohmann_json::nlohmann_json
    pthread
)

add_executable(test_rmsnorm
    ${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm.cpp
)

add_dependencies(test_rmsnorm
    spdlog
)

target_include_directories(test_rmsnorm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_rmsnorm
    coalsack
    pthread
)

target_compile_definitions(test_rmsnorm PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_rope
    ${CMAKE_CURRENT_LIST_DIR}/test_rope.cpp
)

add_dependencies(test_rope
    spdlog
)

target_include_directories(test_rope PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_rope
    coalsack
    pthread
)

target_compile_definitions(test_rope PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_grouped_attention
    ${CMAKE_CURRENT_LIST_DIR}/test_grouped_attention.cpp
)

add_dependencies(test_grouped_attention
    spdlog
)

target_include_directories(test_grouped_attention PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_grouped_attention
    coalsack
    pthread
)

target_compile_definitions(test_grouped_attention PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_moe_router
    ${CMAKE_CURRENT_LIST_DIR}/test_moe_router.cpp
)

add_dependencies(test_moe_router
    spdlog
)

target_include_directories(test_moe_router PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_moe_router
    coalsack
    pthread
)

target_compile_definitions(test_moe_router PUBLIC
    FMT_HEADER_ONLY
)

add_executable(test_expert_mlp
    ${CMAKE_CURRENT_LIST_DIR}/test_expert_mlp.cpp
)

add_dependencies(test_expert_mlp
    spdlog
)

target_include_directories(test_expert_mlp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_expert_mlp
    coalsack
    pthread
)

target_compile_definitions(test_expert_mlp PUBLIC
    FMT_HEADER_ONLY
)
