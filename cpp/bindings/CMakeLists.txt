if(${PYTHON_BINDINGS})
  
  if(${BUILD_WITH_PINOCCHIO_PYTHON})
    set(PINOCCHIO_WITH_PYTHON_INTERFACE true)
  endif()

  # Look for python source directory #################################
  get_filename_component(TOPPRA_PYTHON_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  get_filename_component(TOPPRA_PYTHON_SOURCE_DIR
    ${TOPPRA_PYTHON_SOURCE_DIR} DIRECTORY)
  set(TOPPRA_PYTHON_SOURCE_DIR ${TOPPRA_PYTHON_SOURCE_DIR}/toppra/cpp)
  message(STATUS "Installing Python bindings to ${TOPPRA_PYTHON_SOURCE_DIR}")

  # Build bindings ###################################################
  find_package(Python COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)
  ## Ref: https://pybind11.readthedocs.io/en/stable/cmake/index.html
  message(STATUS "Found ${PYTHON_INCLUDE_DIR} ${PYTHON_EXECUTABLE}")
  pybind11_add_module(toppra_int toppra_int.cpp bindings.cpp
    constraints.cpp paths.cpp)
  target_link_libraries(toppra_int PUBLIC toppra)
  target_include_directories(toppra_int PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(toppra_int PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${TOPPRA_PYTHON_SOURCE_DIR}
    )
  if(BUILD_WITH_PINOCCHIO)
    target_compile_options(toppra_int PRIVATE -DBUILD_WITH_PINOCCHIO)
    target_link_libraries(toppra_int PUBLIC pinocchio::pinocchio)
    if(DEFINED PINOCCHIO_WITH_PYTHON_INTERFACE)
      find_package(Boost REQUIRED COMPONENTS python${PYTHON_VERSION})
      target_compile_options(toppra_int PRIVATE -DPINOCCHIO_WITH_PYTHON_INTERFACE)
      target_link_libraries(toppra_int PUBLIC Boost::python${PYTHON_VERSION})
    endif()
  endif()

endif()
