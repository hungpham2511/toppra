set(SOURCE_FILES
  toppra/constraint.cpp
  toppra/constraint/linear_joint_velocity.cpp
  toppra/constraint/linear_joint_acceleration.cpp
  toppra/constraint/joint_torque.cpp

  toppra/solver.cpp

  toppra/geometric_path.cpp
  toppra/geometric_path/piecewise_poly_path.cpp

  toppra/algorithm/toppra.cpp
  )
if(BUILD_WITH_qpOASES)
    list(APPEND SOURCE_FILES
        toppra/solver/qpOASES-wrapper.cpp)
endif()

add_library(toppra SHARED ${SOURCE_FILES})
target_include_directories(toppra PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${qpOASES_INCLUDE_DIRS})
target_link_libraries(toppra PUBLIC
    ${qpOASES_LIBRARY}
    )

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
  toppra/constraint/linear_joint_velocity.hpp
  toppra/constraint/linear_joint_acceleration.hpp
  toppra/constraint/joint_torque.hpp

  toppra/geometric_path.hpp
  toppra/toppra.hpp

  toppra/solver.cpp
  toppra/solver/qpOASES-wrapper.hpp

  DESTINATION include/toppra)

