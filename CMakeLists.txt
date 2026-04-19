cmake_minimum_required(VERSION 3.8)

project(coalsack)

option(ENABLE_LLM_OPS "Enable LLM/ONNX operations (requires x86 SIMD support)" ON)

find_package(Iconv REQUIRED)
find_package(OpenCV 4.0 REQUIRED)
find_package(Boost REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_RELEASE "-O2")

    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
    endif()
endif()

add_library(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/src/rpc/rpc_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/rpc/rpc_client.cpp
)

if(ENABLE_LLM_OPS)
    target_sources(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/onnx/onnx_proto_parser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/onnx/onnx_importer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/gguf/gguf_loader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/gguf/gguf_multi_loader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/llm/gpt2_tokenizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/llm/chat_template.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/llm/llm_engine.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/llm/moe_weight_provider.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/llm/vision_encoder.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/llm/eagle3_speculative_decoder.cpp
    )
endif()

add_dependencies(${PROJECT_NAME}
    cereal
    spdlog
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

if(ENABLE_LLM_OPS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC
        ENABLE_ONNX_IMPORT
    )
endif()

target_sources(${PROJECT_NAME}
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/rpc/rpc_client.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/rpc/rpc_server.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/rpc/rpc_common.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/data_stream/data_stream_common.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/data_stream/data_stream_receiver.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/data_stream/data_stream_transmitter.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_message.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_edge.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_node.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/subgraph.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_proc.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_proc_server.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_proc_client.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/core/graph_proc_registry.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/image/image.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/image/image_message.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/image/frame_message.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/image/image_nodes.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/gguf/gguf_types.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/gguf/gguf_loader.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/gguf/gguf_multi_loader.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/llm/gpt2_tokenizer.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/llm/chat_template.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/coalsack/llm/llm_engine.h"
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
)

include(${CMAKE_CURRENT_LIST_DIR}/third-party/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sample/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
