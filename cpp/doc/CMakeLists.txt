find_program(DOXYGEN doxygen)
IF(DOXYGEN)
  option(INSTALL_DOCUMENTATION ON "Whether documentation should be installed")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY)

  # Find all the public headers
  set(HEADERS_DIR ${CMAKE_SOURCE_DIR}/src)
  file(GLOB_RECURSE HEADERS ${HEADERS_DIR}/*.hpp ${HEADERS_DIR}/*.cpp)

  set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)
  add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${HEADERS} main.dox
    COMMAND ${DOXYGEN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMENT "Generating documentation")
  if(INSTALL_DOCUMENTATION)
    add_custom_target(doc ALL DEPENDS ${DOXYGEN_INDEX_FILE})

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
      DESTINATION share/${PROJECT_NAME}/doc)
  else()
    add_custom_target(doc DEPENDS ${DOXYGEN_INDEX_FILE})
  endif()
ENDIF()
