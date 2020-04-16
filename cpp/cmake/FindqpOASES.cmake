#
#   Copyright 2019 CNRS
#
#   Author: Guilhem Saurel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

# Try to find qpOASES
# in standard prefixes and in ${qpOASES_PREFIX}
# Once done this will define
#  qpOASES_FOUND - System has qpOASES
#  qpOASES_INCLUDE_DIRS - The qpOASES include directories
#  qpOASES_LIBRARIES - The libraries needed to use qpOASES
#  qpOASES_DEFINITIONS - Compiler switches required for using qpOASES

FIND_PATH(qpOASES_INCLUDE_DIR
  NAMES qpOASES.hpp
  PATHS ${qpOASES_PREFIX} ${qpOASES_PREFIX}/include
  )
FIND_LIBRARY(qpOASES_LIBRARY
  NAMES libqpOASES.so
  PATHS ${qpOASES_PREFIX} ${qpOASES_PREFIX}/lib
  )

SET(qpOASES_LIBRARIES ${qpOASES_LIBRARY})
SET(qpOASES_INCLUDE_DIRS ${qpOASES_INCLUDE_DIR})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(qpOASES DEFAULT_MSG qpOASES_LIBRARY qpOASES_INCLUDE_DIR)
mark_as_advanced(qpOASES_INCLUDE_DIR qpOASES_LIBRARY)
