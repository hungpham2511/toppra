# Try to find GLPK
# in standard prefixes and in ${GLPK_PREFIX}
# Once done this will define
#  GLPK_FOUND - System has GLPK
#  GLPK_INCLUDE_DIRS - The GLPK include directories
#  GLPK_LIBRARIES - The libraries needed to use GLPK
#  GLPK_DEFINITIONS - Compiler switches required for using GLPK

FIND_PATH(GLPK_INCLUDE_DIR
  NAMES glpk.h
  PATHS ${GLPK_PREFIX} ${GLPK_PREFIX}/include
  DOC "GLPK include directory")
FIND_LIBRARY(GLPK_LIBRARY
  NAMES glpk
  PATHS ${GLPK_PREFIX} ${GLPK_PREFIX}/lib
  DOC "GLPK library")

SET(GLPK_LIBRARIES ${GLPK_LIBRARY})
SET(GLPK_INCLUDE_DIRS ${GLPK_INCLUDE_DIR})

# TODO Version could be extracted from the header.

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLPK DEFAULT_MSG GLPK_LIBRARY GLPK_INCLUDE_DIR)
mark_as_advanced(GLPK_INCLUDE_DIR GLPK_LIBRARY)
