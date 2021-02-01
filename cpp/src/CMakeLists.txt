include(GenerateExportHeader)

set(SOURCE_FILES
  toppra/constraint.cpp
  toppra/constraint/linear_joint_velocity.cpp
  toppra/constraint/linear_joint_acceleration.cpp
  toppra/constraint/joint_torque.cpp
  toppra/constraint/cartesian_velocity_norm.cpp

  toppra/solver.cpp
  toppra/solver/seidel.cpp

  toppra/geometric_path.cpp
  toppra/geometric_path/piecewise_poly_path.cpp
  toppra/parametrizer/const_accel.cpp
  toppra/parametrizer.cpp

  toppra/algorithm.cpp
  toppra/algorithm/toppra.cpp
  )
if(BUILD_WITH_qpOASES)
    list(APPEND SOURCE_FILES
        toppra/solver/qpOASES-wrapper.cpp)
endif()
if(BUILD_WITH_GLPK)
    list(APPEND SOURCE_FILES
        toppra/solver/glpk-wrapper.cpp)
endif()

add_library(toppra SHARED ${SOURCE_FILES})
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/toppra)
generate_export_header(toppra
  EXPORT_FILE_NAME toppra/export.hpp
  #INCLUDE_GUARD_NAME TOPPRA_EXPORT_HPP # CMake >= 3.11 only
  )
target_include_directories(toppra PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${qpOASES_INCLUDE_DIRS}
  ${GLPK_INCLUDE_DIRS}
  )
foreach(solver qpOASES GLPK)
  if(BUILD_WITH_${solver})
    target_compile_definitions(toppra PRIVATE -DBUILD_WITH_${solver})
    target_link_libraries(toppra PUBLIC ${${solver}_LIBRARIES})
    target_include_directories(toppra PUBLIC ${${solver}_INCLUDE_DIRS})
  endif()
endforeach()
if (TOPPRA_DEBUG_ON)
  target_compile_definitions(toppra PRIVATE TOPPRA_DEBUG_ON)
endif()
if (TOPPRA_WARN_ON)
  target_compile_definitions(toppra PRIVATE TOPPRA_WARN_ON)
endif()
if (OPT_MSGPACK)
  target_compile_definitions(toppra PRIVATE TOPPRA_OPT_MSGPACK)
endif()

if (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3) 
  target_compile_definitions     (toppra PUBLIC ${EIGEN3_DEFINITIONS} ) 
  target_include_directories (toppra PUBLIC ${EIGEN3_INCLUDE_DIRS} ) 
else (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3) 
  target_link_libraries(toppra PUBLIC Eigen3::Eigen) 
endif (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3) 

install(TARGETS toppra EXPORT toppra::toppra
  DESTINATION lib)
install(FILES
  toppra/algorithm.hpp
  toppra/constraint.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/toppra/export.hpp

  toppra/geometric_path.hpp
  toppra/parametrizer.hpp
  toppra/toppra.hpp

  toppra/solver.hpp

  DESTINATION include/toppra)
install(FILES
  toppra/algorithm/toppra.hpp
  DESTINATION include/toppra/algorithm)
install(FILES
  toppra/constraint/linear_joint_velocity.hpp
  toppra/constraint/linear_joint_acceleration.hpp
  toppra/constraint/joint_torque.hpp
  toppra/constraint/cartesian_velocity_norm.hpp
  DESTINATION include/toppra/constraint)
install(FILES
  toppra/geometric_path/piecewise_poly_path.hpp
  DESTINATION include/toppra/geometric_path)
install(FILES
  toppra/parametrizer/const_accel.hpp
  DESTINATION include/toppra/parametrizer)
install(FILES
  toppra/constraint/joint_torque/pinocchio.hpp
  DESTINATION include/toppra/constraint/joint_torque)
install(FILES
  toppra/constraint/cartesian_velocity_norm/pinocchio.hpp
  DESTINATION include/toppra/constraint/cartesian_velocity_norm)
install(FILES
  toppra/solver/qpOASES-wrapper.hpp
  toppra/solver/glpk-wrapper.hpp
  toppra/solver/seidel.hpp
  DESTINATION include/toppra/solver)
