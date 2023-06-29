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