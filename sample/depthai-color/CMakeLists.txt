add_executable(depthai_color
    ${CMAKE_CURRENT_LIST_DIR}/depthai_color.cpp
)

add_dependencies(depthai_color
    cereal
    spdlog
    libjpeg-turbo
    json
)

target_include_directories(depthai_color PUBLIC
    $<$<BOOL:${ENABLE_DEPTHAI_EXT}>:/usr/local/include>
)

target_link_libraries(depthai_color
    ${OpenCV_LIBS}
    coalsack
    pthread
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib/libturbojpeg.a
    ${CMAKE_BINARY_DIR}/lz4/lib/liblz4.a
    $<$<BOOL:${ENABLE_DEPTHAI_EXT}>:depthai::core>
    $<$<BOOL:${ENABLE_DEPTHAI_EXT}>:depthai::opencv>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${ENABLE_DEPTHAI_EXT}>:ENABLE_DEPTHAI_EXT>
)