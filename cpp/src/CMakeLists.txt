add_library(toppra SHARED
  toppra/constraint.cpp
  toppra/constraint/linear_joint_velocity.cpp
  toppra/constraint/linear_joint_acceleration.cpp

  toppra/geometric_path.cpp
  toppra/geometric_path/piecewise_poly_path.cpp
  )

target_include_directories(toppra PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
 
if (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3) 
  target_compile_definitions     (toppra PUBLIC ${EIGEN3_DEFINITIONS} ) 
  target_include_directories (toppra PUBLIC ${EIGEN3_INCLUDE_DIRS} ) 
else (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3) 
  target_link_libraries(toppra PUBLIC Eigen3::Eigen3) 
endif (${EIGEN3_VERSION_STRING} VERSION_LESS 3.3) 


install(TARGETS toppra EXPORT toppra::toppra
  DESTINATION lib)
install(FILES
  toppra/algorithm.hpp

  toppra/constraint.hpp
  toppra/constraint/linear_joint_velocity.hpp
  toppra/constraint/linear_joint_acceleration.hpp

  toppra/geometric_path.hpp
  toppra/toppra.hpp
  DESTINATION include/toppra)

