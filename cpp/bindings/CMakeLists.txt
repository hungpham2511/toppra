if(${PYTHON_BINDINGS})

  # Look for python source directory #################################
  get_filename_component(TOPPRA_PYTHON_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  get_filename_component(TOPPRA_PYTHON_SOURCE_DIR
    ${TOPPRA_PYTHON_SOURCE_DIR} DIRECTORY)
  set(TOPPRA_PYTHON_SOURCE_DIR ${TOPPRA_PYTHON_SOURCE_DIR}/toppra/cpp)
  message(STATUS "Installing Python bindings to ${TOPPRA_PYTHON_SOURCE_DIR}")

  # Build bindings ###################################################
  find_package(pybind11 REQUIRED)
  message(STATUS "Found ${PYTHON_INCLUDE_DIR} ${PYTHON_EXECUTABLE}")
  pybind11_add_module(toppra_int toppra_int.cpp bindings.cpp
    constraints.cpp)
  target_link_libraries(toppra_int PUBLIC toppra)
  target_include_directories(toppra_int PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(toppra_int PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${TOPPRA_PYTHON_SOURCE_DIR}
    )
  if(BUILD_WITH_PINOCCHIO)
    target_compile_options(toppra_int PRIVATE -DBUILD_WITH_PINOCCHIO)
    target_link_libraries(toppra_int PUBLIC pinocchio::pinocchio)
    if(DEFINED PINOCCHIO_WITH_PYTHON_INTERFACE)
      find_package(Boost REQUIRED COMPONENTS python3)
      target_compile_options(toppra_int PRIVATE -DPINOCCHIO_WITH_PYTHON_INTERFACE)
      target_link_libraries(toppra_int PUBLIC Boost::python3)
    endif()
  endif()

endif()
