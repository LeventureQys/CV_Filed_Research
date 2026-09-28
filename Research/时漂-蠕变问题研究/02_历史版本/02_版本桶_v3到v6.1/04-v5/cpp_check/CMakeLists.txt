cmake_minimum_required(VERSION 3.20)
project(v5_cpp_check LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
add_executable(v5_cpp_check
    main.cpp
    "${ROOT}/src/domain/drift/drift_compensator.cpp")
target_include_directories(v5_cpp_check PRIVATE
    "${ROOT}/src"
    "${ROOT}/thirdparty/eigen-5.0.0")
if (MSVC)
    target_compile_options(v5_cpp_check PRIVATE /EHsc /utf-8)
endif()
