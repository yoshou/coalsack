project(tests)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/syncer_test.cpp
)

add_dependencies(${PROJECT_NAME}
    googletest
    spdlog
)

include_directories(
    SYSTEM ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest.a
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest_main.a
    pthread
)