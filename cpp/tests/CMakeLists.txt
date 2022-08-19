
# Now simply link against gtest or gtest_main as needed. Eg
set(TEST_SOURCES
  gtest_main.cpp
  test_constraints.cpp
  test_solver.cpp
  test_poly_geometric_path.cpp
  test_hermite_spline.cpp
  test_cubic_spline.cpp
  test_algorithm.cpp
  test_parametrizer.cpp
  test_spline_parametrizer.cpp
  test_toppra_approach.cpp

  test_issue_198.cpp
  )

add_executable(all_tests ${TEST_SOURCES})
target_link_libraries(all_tests PUBLIC toppra gtest ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(all_tests PRIVATE .)

if(BUILD_WITH_PINOCCHIO)
  target_compile_definitions(all_tests PRIVATE -DBUILD_WITH_PINOCCHIO)
  target_link_libraries(all_tests PUBLIC pinocchio::pinocchio)
endif()
if(BUILD_WITH_qpOASES)
  target_compile_definitions(all_tests PRIVATE -DBUILD_WITH_qpOASES)
endif()
if(BUILD_WITH_GLPK)
  target_compile_definitions(all_tests PRIVATE -DBUILD_WITH_GLPK)
endif()
if (OPT_MSGPACK)
  target_compile_definitions(all_tests PRIVATE -DTOPPRA_OPT_MSGPACK)
endif()
if (TOPPRA_DEBUG_ON)
  target_compile_definitions(all_tests PRIVATE -DTOPPRA_DEBUG_ON)
endif()
if (TOPPRA_WARN_ON)
  target_compile_definitions(all_tests PRIVATE TOPPRA_WARN_ON)
endif()
add_test(NAME all_tests COMMAND all_tests)
