cmake_minimum_required(VERSION 3.0.0)
project(perf_test_concurrent_queues VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-interference-size -fconcepts")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

add_executable(perf_test_concurrent_queues bin/perf_test_concurrent_queues.cpp)
target_include_directories(perf_test_concurrent_queues PUBLIC "include/")
target_compile_features(perf_test_concurrent_queues PUBLIC cxx_std_23)

add_executable(multicast_ringbuff_unittest bin/multicast_ringbuff_unittest.cpp)
target_include_directories(multicast_ringbuff_unittest PUBLIC "include/")
target_compile_features(multicast_ringbuff_unittest PUBLIC cxx_std_23)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
