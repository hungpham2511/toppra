cmake_minimum_required(VERSION 3.5)

project(toppra)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Dynamics
option(BUILD_WITH_PINOCCHIO "Compile with Pinocchio library" OFF)
# Solvers
option(BUILD_WITH_qpOASES "Compile the wrapper for qpOASES" OFF)
option(BUILD_WITH_GLPK "Compile the wrapper for GLPK (GPL license)" OFF)
# General options
option(TOPPRA_DEBUG_ON "Set debug mode to on." OFF)

find_package (Threads)

find_package (Eigen3 REQUIRED)
message(STATUS "Found Eigen version ${EIGEN3_VERSION_STRING}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
if(BUILD_WITH_PINOCCHIO)
    find_package(pinocchio REQUIRED)
    message(STATUS "Found pinocchio ${pinocchio_VERSION}")
endif()
if(BUILD_WITH_qpOASES)
    find_package(qpOASES REQUIRED)
    message(STATUS "Found qpOASES")
endif()
if(BUILD_WITH_GLPK)
    find_package(GLPK REQUIRED)
    message(STATUS "Found glpk ${GLPK_LIBRARY} ${GLPK_INCLUDE_DIR}")
endif(BUILD_WITH_GLPK)

add_subdirectory(src)
add_subdirectory(doc)

# Download and unpack googletest at configure time
# source: https://github.com/google/googletest/tree/master/googletest
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

enable_testing()
add_subdirectory(tests)
