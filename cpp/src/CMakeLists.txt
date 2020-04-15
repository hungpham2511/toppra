add_library(toppra SHARED
  toppra/constraint.cpp
  toppra/constraint/linear_joint_velocity.cpp
  )
target_include_directories(toppra PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(toppra PUBLIC Eigen3::Eigen)

install(TARGETS toppra EXPORT toppra::toppra
  DESTINATION lib)
install(FILES
  toppra/algorithm.hpp

  toppra/constraint.hpp
  toppra/constraint/linear_joint_velocity.hpp

  toppra/geometric_path.hpp
  toppra/toppra.hpp
  DESTINATION include/toppra)
