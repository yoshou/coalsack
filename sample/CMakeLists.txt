cmake_minimum_required(VERSION 3.8)

project(graph_proc_server)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/server.cpp
)

add_dependencies(${PROJECT_NAME}
    cereal
    spdlog
)

target_link_libraries(${PROJECT_NAME} coalsack pthread boost_system boost_thread)

project(graph_proc_client)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/client.cpp
)

add_dependencies(${PROJECT_NAME}
    cereal
    spdlog
)

target_link_libraries(${PROJECT_NAME} coalsack pthread boost_system boost_thread)
