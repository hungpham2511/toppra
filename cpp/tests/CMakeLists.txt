
# Now simply link against gtest or gtest_main as needed. Eg
set(TEST_SOURCES
  gtest_main.cpp
  test_constraints.cpp
  test_solver.cpp
  test_poly_geometric_path.cpp
  )

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests PUBLIC toppra gmock gtest ${CMAKE_THREAD_LIBS_INIT})

if(BUILD_WITH_PINOCCHIO)
  target_compile_definitions(all_tests PRIVATE -DBUILD_WITH_PINOCCHIO)
  target_link_libraries(all_tests PUBLIC pinocchio::pinocchio)
endif()
if(BUILD_WITH_qpOASES)
  target_compile_definitions(all_tests PRIVATE -DBUILD_WITH_qpOASES)
endif()
