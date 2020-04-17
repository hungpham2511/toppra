
# Now simply link against gtest or gtest_main as needed. Eg
set(TEST_SOURCES
  gtest_main.cpp
  constraints.cpp
  test_poly_geometric_path.cpp
  )

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests toppra gmock gtest ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(tests PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )
