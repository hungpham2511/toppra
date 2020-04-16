set(SOURCE_FILES
  toppra/constraint.cpp
  toppra/constraint/linear_joint_velocity.cpp
  toppra/constraint/linear_joint_acceleration.cpp
  toppra/constraint/joint_torque.cpp

  toppra/solver.cpp
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
    Eigen3::Eigen
    ${qpOASES_LIBRARY}
    )

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
