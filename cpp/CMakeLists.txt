cmake_minimum_required(VERSION 3.10)

project(toppra)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package (Threads)


set(GMOCK_HOME extern/googletest)


include_directories(
  ${GMOCK_HOME}/googlemock/include
  ${GMOCK_HOME}/googletest/include
  )

link_directories(
  ${GMOCK_HOME}/mybuild/lib
  )


set(TEST_SOURCES
  tests/gtest_main.cpp
  )

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests gmock gtest ${CMAKE_THREAD_LIBS_INIT})

