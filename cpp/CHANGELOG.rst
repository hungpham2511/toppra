^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package toppra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.6.7 (2026-04-09)
------------------
* build(cpp): Modernize googletest CMake usage (`#283 <https://github.com/hungpham2511/toppra/issues/283>`_)
* Contributors: Sebastian Castro

0.6.6 (2026-03-31)
------------------
* [cpp] Add missing cassert includes (`#280 <https://github.com/hungpham2511/toppra/issues/280>`_)
* [cpp] Fix CMake syntax for Eigen3 version check (`#279 <https://github.com/hungpham2511/toppra/issues/279>`_)
* [ros] Add python3-dev and eigen as a dependency in package.xml (`#278 <https://github.com/hungpham2511/toppra/issues/278>`_)
* Contributors: Sebastian Castro

0.6.5 (2026-03-27)
------------------
* [python]: Go back to master branch of hungpham2511 qpOASES repo (`#273 <https://github.com/hungpham2511/toppra/issues/273>`_)
* Prepare for ROS build farm (`#277 <https://github.com/hungpham2511/toppra/issues/277>`_)
* Contributors: Sebastian Castro

0.6.4 (2026-01-26)
------------------
* chore: Bump version to 0.6.4 (`#268 <https://github.com/hungpham2511/toppra/issues/268>`_)
* Fix Python and C++ CI, disable ECOS solver in tests (`#270 <https://github.com/hungpham2511/toppra/issues/270>`_)
* build: Move Python library to its own subfolder to support building as ROS 2 package (`#266 <https://github.com/hungpham2511/toppra/issues/266>`_)
* fix(test): Use ASSERT_NEAR for 0 comparison in test_parametrizer.cpp (`#265 <https://github.com/hungpham2511/toppra/issues/265>`_)
* fix(cpp): incoherent bounds seidel lp 1d (`#247 <https://github.com/hungpham2511/toppra/issues/247>`_)
* Contributors: Erik Holum, Joseph Mirabel, Maxim Skripnik, Sebastian Castro, Silvio Traversaro

0.6.3 (2025-03-16)
------------------
* Updated interface for numpy > 2.0 (`#259 <https://github.com/hungpham2511/toppra/pull/259>`_)
* Contributors: Zachary Kingston

0.6.2 (2023-09-19)
------------------
* chore: Bump toppra version: '0.6.1' --> '0.6.2'
* feat: warn when solver fails (`#241 <https://github.com/hungpham2511/toppra/issues/241>`_)
* Release version 0.6.2 (`#242 <https://github.com/hungpham2511/toppra/issues/242>`_)
* Contributors: Hung Pham (Phạm Tiến Hùng), Joseph Mirabel

0.6.1 (2023-04-21)
------------------
* chore: Bump toppra version: '0.6.0' --> '0.6.1' ( `#233 <https://github.com/hungpham2511/toppra/issues/233>`_)
* fix: fix numerical issue in Seidel 1D solver.
* Contributors: Joseph Mirabel

0.6.0 (2023-03-08)
------------------
* chore: Bump toppra version: '0.5.2' --> '0.6.0'
* feat(cpp): Add initial gridpoints to proposeGridpoints (`#227 <https://github.com/hungpham2511/toppra/issues/227>`_)
* feat(cpp,constraint): Make Interpolation the default method (`#225 <https://github.com/hungpham2511/toppra/issues/225>`_)
* Contributors: Hung, Hung Pham (Phạm Tiến Hùng), Joseph Mirabel

0.5.2 (2022-11-19)
------------------
* Bump version: 0.5.1 → 0.5.2
* chore: make C++ always define all installed symbols. (`#215 <https://github.com/hungpham2511/toppra/issues/215>`_)
* feat: Allow setting N=0 to select gridpoints automatically (`#201 <https://github.com/hungpham2511/toppra/issues/201>`_)
* feat: Improve the cubic spline, hermite spline factory methods (`#202 <https://github.com/hungpham2511/toppra/issues/202>`_)
* doc,build: Minor adjustments to docstring for a clearer explaination
* doc: Improve PiecewisePolyPath constructor docstring
* Contributors: Hung, Hung Pham (Phạm Tiến Hùng), Joseph Mirabel

0.5.1 (2022-07-15)
------------------
* Bump version: 0.5.0 → 0.5.1
* Contributors: Hung

0.5.0 (2022-07-14)
------------------
* Bump version: 0.4.2 → 0.5.0
* Make path times public. (`#197 <https://github.com/hungpham2511/toppra/issues/197>`_)
* Fixed bug in ConstAccel parametrizer where evaluating valid times can sometimes result in an exception being thrown. (`#193 <https://github.com/hungpham2511/toppra/issues/193>`_)
* Google Test: master -> main (`#192 <https://github.com/hungpham2511/toppra/issues/192>`_)
* CI works (`#191 <https://github.com/hungpham2511/toppra/issues/191>`_)
* doc: Add some doc to compiling C++
* Fixed incorrect variable bug in seidel.cpp which was causing memory issues. (`#190 <https://github.com/hungpham2511/toppra/issues/190>`_)
* Added a check to test the input times to the constaccel parametrizer are within the bounds of the parameterizer, and clampe resulting path values to the lower bound as well as the upper bound.
* Chnaged path length in test to a more "usual" length value to avoid condusion
* Clamp all path times to the upper bounds of the path in the const accel parametrizer. Fixes `#184 <https://github.com/hungpham2511/toppra/issues/184>`_.
* Allow to build static library (`#178 <https://github.com/hungpham2511/toppra/issues/178>`_)
* Add BUILD_TESTS option (`#181 <https://github.com/hungpham2511/toppra/issues/181>`_)
* Fixes for windows builds (`#179 <https://github.com/hungpham2511/toppra/issues/179>`_)
* Contributors: Antoine Hoarau, Hung, Hung Pham (Phạm Tiến Hùng), Jason Ernst, Jess Moss, Joseph Mirabel, Steve Golton

0.4.2 (2021-08-08)
------------------
* Bump version: 0.4.1 → 0.4.2
* Replace for loop in piecewise poly path to find index by std::lower_bound. (`#177 <https://github.com/hungpham2511/toppra/issues/177>`_)
* Remove invalid characters from README.md (`#173 <https://github.com/hungpham2511/toppra/pull/173>`_)
* [CPP] Implement Spline Parametrizer (`#162 <https://github.com/hungpham2511/toppra/issues/162>`_)
* Bump version: 0.4.0 → 0.4.1
* Contributors: Hung, Hung Pham (Phạm Tiến Hùng), Jakob LUDWIGER, John Wason, leonardoedgar

0.4.0 (2021-03-16)
------------------
* Bump version: 0.3.1 → 0.4.0
* Merge branch 'develop'
* [ci]Fix version. Add bump2version config file to manage.
* [ci,cmake]Missing python version in cmake command
* [ci,cmake]Find boost with given python version
* Merge pull request `#161 <https://github.com/hungpham2511/toppra/issues/161>`_ from leonardoedgar/develop
  [cpp] Construct spline with 1st and 2nd order derivatives at curve endpoints
* add docstring and minor changes based on comments
* remove unnecessary variables, copy constructors, and minor fixes based on comments
* add support for mixed 1st and 2nd order boundary conditions
* move duplicated function makeVectors into a separate file
* add tests for constructing spline from 1st and 2nd order derivatives at curve endpoints
* implement spline construction from 1st and 2nd order derivatives from the curve endpoints
* Merge pull request `#160 <https://github.com/hungpham2511/toppra/issues/160>`_ from jmirabel/develop
  [cpp] Enhance Seidel solver.
* [cpp][seidel] Comment and clarify code
* [cpp][seidel] Move re-ordering of rows in Seidel::solveStagewiseOptim
* [cpp][Seidel] Remove solve_lp1d_atomic
* [cpp] fix handling of THR_VIOLATION in Seidel.
* [cpp] Fix bugs.
* [cpp] Add PathParametrizationAlgorithm::setInitialXBounds.
* [Seidel] Robustify and improve performances + add unit test from real experiments.
* [Seidel] Improve logging macro.
* [Seidel] Fix prototypes.
* [Seidel] Better handling of bound constraints.
* [Seidel] Improve numerical stability + unit test seidel functions.
* Add package.xml to allow this package to build with catkin.
* Add TOPPRA_LOG_WARN.
* Merge pull request `#158 <https://github.com/hungpham2511/toppra/issues/158>`_ from jmirabel/develop
  [cpp] Add Seidel solver.
* [C++] Run algo unit test for each solvers.
* [C++] Use Seidel if no other solver are available.
* [C++] Seidel: cosmetic change.
* [C++] Seidel: cosmetic change.
* [C++] Simplify test_solver.cpp
* [C++] Simplification of Seidel solver.
* [C++][Minor] Fix debug output.
* [C++] Add toppra::solver::Seidel
* [C++] PathParametrizationAlgorithm::getParameterizationData returns a const ref
* [Doc] Add comment (related to `#156 <https://github.com/hungpham2511/toppra/issues/156>`_)
* Merge pull request `#153 <https://github.com/hungpham2511/toppra/issues/153>`_ from Synxis/develop
  Fix wrong variable for velocity
* Fix wrong variable velocity used
* Merge pull request `#146 <https://github.com/hungpham2511/toppra/issues/146>`_ from jmirabel/develop
  [cpp] Add CartesianVelocityNorm and an implemention with Pinocchio
* [cpp][doc] Fix inheritance missed by Doxygen.
* [cpp] Expose CartesianVelocityNorm in Python
* [cpp] Add CartesianVelocityNorm and an implemention with Pinocchio
* Merge pull request `#143 <https://github.com/hungpham2511/toppra/issues/143>`_ from jmirabel/develop
  [Cpp] Add PathParametrizationAlgorithm::setGridpoints
* [Cpp] Add PathParametrizationAlgorithm::setGridpoints
* Merge pull request `#129 <https://github.com/hungpham2511/toppra/issues/129>`_ from hungpham2511/feat-parametrizer
  [cpp]Implement trajectory parametrizer
* [cpp]Update const_accel code based on comment
* [cpp]Add a minor comment on TOPPRA_NEARLY_ZERO
* [cpp]Add missing file. Modify code based on review. Add unit test.
* Update HISTORY.md. Remove unused files.
* [cpp]Minor docstring improvement.
* [cpp]Improve test suite. Disable unhelpful assertions.
* [cpp]Tidy up const accel parametrizer code. Fix bug.
* [cpp]Turn on DEBUG mode in tests
* [cpp]Add test to constaccel parametrizer. Bug fixed.
* [cpp]Bug fixed
* Merge remote-tracking branch 'origin/feat-parametrizer' into feat-parametrizer
* Merge pull request `#140 <https://github.com/hungpham2511/toppra/issues/140>`_ from ndehio/feat-parametrizer
  Adding full cpp-example using parametrizer
* [docs]Improve some docstrings
* Minor bugs fixed
* [docs]Add instruction for building doxygen doc
* [docs]Add some instructions to building cpp code
* add full toppra-example (note: test fails...)
* add missing parametrizer in CMake
* fix issue: assign return-code in m_data.ret_code
* add missing validate() method
* [cpp]Implement reparametrization internal computation
* [cpp]Reorganize API design. Use less virtual public methods.
* [cpp]Reformat code with clang
* [cpp]Compute time instances and accels from data
* [cpp]Setup the overall API
* Contributors: Hung, Hung Pham (Phạm Tiến Hùng), Joseph Mirabel, Niels Dehio, Thibault LESCOAT, leonardoedgar

0.3.1 (2020-08-23)
------------------
* Release v0.3.1
* Merge pull request `#125 <https://github.com/hungpham2511/toppra/issues/125>`_ from jmirabel/develop
  [cpp] Add the possible to use varying joint velocity limits.
* [cpp] Add the possible to use varying joint velocity limits.
* Merge branch 'develop' into feat/striptype
* Merge pull request `#111 <https://github.com/hungpham2511/toppra/issues/111>`_ from jmirabel/develop
  [C++] Fix GLPK bound type + numerical issues
* [cpp] Initialiaze m_configSize in PiecewisePolyPath::initAsHermite
* [cpp] Add some getter and setter to JointTorque
* [cpp] Check boundaries in solver + fix numerical issue.
* [C++] Add some debugging message.
* [C++] Fix GLPK bound type.
* Merge pull request `#120 <https://github.com/hungpham2511/toppra/issues/120>`_ from hungpham2511/bindings
  [cpp]A numerical improvement. Refine tests.
* [cpp]Improve numerical evaluation for qpoases solver
* [cpp]Clamp velocities to be within the controllable sets
* [cpp]Add an option to force build python bindings
* Merge pull request `#118 <https://github.com/hungpham2511/toppra/issues/118>`_ from jmirabel/bindings
  [cpp][cmake] Fix PYTHON_VERSION variable.
* Merge pull request `#119 <https://github.com/hungpham2511/toppra/issues/119>`_ from hungpham2511/feat/bindings
  [cpp]Import pinocchio only when PINOCCHIO python flag is defined
* [cpp]Import pinocchio only when PINOCCHIO python flag is defined
* [cpp][cmake] Fix PYTHON_VERSION variable.
* Merge branch 'develop' into feat/striptype
* Merge pull request `#114 <https://github.com/hungpham2511/toppra/issues/114>`_ from hungpham2511/feat/bindings
  [cpp]Feature: Bindings for multiple components
* Merge branch 'develop' into feat/bindings
* Merge pull request `#116 <https://github.com/hungpham2511/toppra/issues/116>`_ from jmirabel/bindings
  [cpp] Update Python Bindings + fix numerical issues.
* [cpp] Fix binding of LinearConstraint::computeParams + test.
* [cpp][minor] Update Python bindings.
* [cpp] Allow friction to be null + Fix jointTorque::Pinocchio memory management
* Merge pull request `#115 <https://github.com/hungpham2511/toppra/issues/115>`_ from hungpham2511/python2
  [cpp]Add an option to configure Python version when building the Python bindings
* [Minor] Fix documentation.
* [cpp] Fix numerical issue (tiny negative sd^2)
* [cpp] Expose GeometricPath.computeParams to Python.
* [cpp] Fix collocationToInterpolate
* [cpp] move bindings of geometric paths into separate file.
* [cpp] move bindings of constraints into separate file.
* [cpp] Add jointTorque::Pinocchio::fromURDF
* [cpp] Remove unnecessary wrappers.
* [cpp] Python bindings directly use the C++ interface.
* [cpp] Use shared ptr for LinearConstraint derived classes.
* [cpp] Provide bindings for jointTorque::Pinocchio
* [cpp] Fix Python bindings.
  in order to make them fully compatible with current Python
  implementation of TOPP-RA.
* [cpp]Fix a numerical issue
* [cpp]Provide option to build pybind against different python versions
* [cpp]Bind toppra algorithm.
* [cpp]Bind supporting structs
* [cpp]Reformat code
* [cpp]Bind accel constraints
* [cpp]Linear Velocity constraint
* Merge pull request `#110 <https://github.com/hungpham2511/toppra/issues/110>`_ from hungpham2511/feat/bindings
  [cpp,python]Feature: Improve PiecewisePolyPath
* [cpp]constructHermite as static method
* format code, fix compilation issue
* [ci]Fix CI to install msgpack
* [binding] Provide binding for constructHermite
* [cpp] Construct a Hermite Cubic spline as piecewise poly
* Add test case for hermite spline.
* [python] Bind serialization to python
* [cpp] Make serialization optional with OPT_MSGPACK option
* [cpp] Implement serialization for trajectory class using msgpack
* Use const ref to avoid copy data
* Extract piecewise_poly_path declaration to a separate header
* Implement conversion of Matrices, bind path_interval
* explicitly cast to py::array_t
* Add test case for python bindings.
* Install pybind11 bindings to python src dir.
* Initial bindings
* Merge pull request `#99 <https://github.com/hungpham2511/toppra/issues/99>`_ from jmirabel/refactoring_solver
  Update API to enable user to choose a solver.
* Merge branch 'develop' into refactoring_solver
* [Minor] Fix include order.
* [Minor] Fix some doc.
* Move default Solver creation to Solver and initialization to TOPPRA.
* Select default solver according to build options.
* Update unit test to refactoring.
* Adapt PathParametrizationAlgorithm to refactoring of Solver.
* Refactor Solver API.
* Contributors: Hung, Hung Pham (Phạm Tiến Hùng), Joseph Mirabel

0.3.0 (2020-05-03 22:13:02 +0800)
---------------------------------
* Merge pull request `#100 <https://github.com/hungpham2511/toppra/issues/100>`_ from hungpham2511/release-v0.3.0
  Release v0.3.0
* Bump version for C++ API
* Merge pull request `#97 <https://github.com/hungpham2511/toppra/issues/97>`_ from jmirabel/develop
  Add solver based on GLPK.
* [cpp] Update version + update readme.
* Fix CMake < 3.5 + some doc + declaration of infty.
* Minor update of the documentation.
* Merge remote-tracking branch 'origin/develop' into develop
* Merge pull request `#90 <https://github.com/hungpham2511/toppra/issues/90>`_ from hungpham2511/feat-cpp
  Feature: TOPPRA algorithm
* Update cpp/src/toppra/algorithm/toppra.cpp
  Co-Authored-By: Joseph Mirabel <jmirabel@laas.fr>
* Some minor clean ups.
* Implement compute feasible function
* Add missing installation of header
* Use value_type instead of double, where relevant.
* Change template parameter of jointTorque::Pinocchio.
* Modify code based on code reviews.
* [CMake] Add config, version and target files.
* Fix install + documentation
* Add export macros.
* [CMake] install documentation.
* [CMake] Add missing headers.
* Remove unnecessary includes.
* Merge remote-tracking branch 'origin/feat-cpp' into tmp2
* Use move semantics instead of const ref.
* Reformat code with clang-format.
* Add several debug messages and option to turn off
* Fix bug in algorithm and relax test criteria.
* Fix bug in test generation.
* Add missing #ifdef..endif in unit test.
* Add test that checks solvers consistency.
* Add solver::GLPKWrapper
* Fix conflicts after merge.
* Merge branch 'develop' into feat-cpp
* Merge pull request `#94 <https://github.com/hungpham2511/toppra/issues/94>`_ from jmirabel/develop
  Fix bug in qpOASESWrapper.
* Add two more test cases to check for controllable and parametrization.
* Fix qpOASESWrapper when i == N.
* Fix test_poly_geometric_path.cpp
* Add test for the full algorithm with PYthon code.
* Add another test for geometric path comparing results with scipy.
* Fix bug in solver wrapper initialization.
* Add a simple logging method with macro.
* Update changelog
* Change index type from int to size_t.
* Fix bug in qpOASESWrapper.
* Disable failing test (temporary).
* Format code with clang.
* Update function docstring.
* Shift algorithm data to data struct.
* Implement TOPPRA forward pass.
* Commit missing file.
* Use return code in the parametrization algorithm.
* Bug in polynomial path fixed.
* public API for allgorithm base class
* Minor changes based on review.
* Setup the basic public interface.
* Add a failing test showing usage.
* Include compilation instruction.
* Merge pull request `#86 <https://github.com/hungpham2511/toppra/issues/86>`_ from jmirabel/develop
  Add Solver + solver::qpOASESWrapper + JointTorque + jointTorque::Pinocchio
* Make qpOASES box bound on variable modifyable.
* Make LinearConstraint::allocateParams not mandatory.
* [Doc] Add a brief main page.
* [Doc] enable EXTRACT_PRIVATE
* Fix test_poly_geometric_path.cpp
* Fix unit test using Pinocchio
* Fix cpp/cmake/FindqpOASES.cmake
* Merge branch 'develop' into feat-traj-quality
* Enable `make test`
* Add a todo.
* Use std::unique_ptr
* Rename class attributes
* Prepend test\_ to test files
* Merge pull request `#88 <https://github.com/hungpham2511/toppra/issues/88>`_ from hungpham2511/ci/github-action
  Setup github action for linting C++ codebase
* Fix API of computeParams and computeParams_impl
* Improve tests
* Update unit-test to use PiecewisePolyPath.
* Use GeometricPath in constraints.
* [GeometricPath] Add config size + store dimension in abstract class + doc.
* [GeometricPath] Add missing const + virtual destructor.
* [CMake] Remove unnecessary line.
* Merge remote-tracking branch 'origin/develop' into develop
* Fix typo in CMake file.
* Merge pull request `#87 <https://github.com/hungpham2511/toppra/issues/87>`_ from hungpham2511/feat-cpp
  Add CI configure to check C++ codebase
* Configure CMake to work with eigen < 3.3
* Downgrade CMAke required version.
* Merge pull request `#85 <https://github.com/hungpham2511/toppra/issues/85>`_ from hungpham2511/feat-cpp
  Feature: GeometricPath and PiecewisePolyPath (WIP)
* Add missing source file.
* Minor.
* Provide a default implementation for GeometricPath.
* Fix abstract interface.
* Add virtual methods to base class.
* Reorganize and refactor piecewise poly.
* Implement a test case to profile poly evaluation.
* Clean up CMakeList.txt
* Separate source / implementation.
* Fix compilation with optional dependencies
* Add cmake/FindqpOASES.cmake
* Add unit-test for qpOASESWrapper
* Add implementation of JointTorque using Pinocchio.
* Handle friction in JointTorque
* Add virtual desctrutor and make constructor protected.
* Add JointTorque constraint.
* Implement ppoly evaluation.
* Add linking to cmake list.
* Merge pull request `#83 <https://github.com/hungpham2511/toppra/issues/83>`_ from jmirabel/develop
  First draft of LinearConstraint
* Update Solver to removal of BoxConstraint
* Merge branch 'develop' into solver
* Fix handling of Interpolation case.
* Remove class BoxConstraint
* Add first version of solver::qpOASESWrapper
* Make some attribute of Solver protected.
* Add missing virtual destructors.
* Add base class Solver.
* Add forward declarations.
* Add LinearJointAcceleration
* Handle Interpolation discretization type.
* Rework print functions.
* When Constraint::constantF, both F and g are constant.
* Add basic test for LinearJointVelocity
* Add LinearJointVelocity
* Add missing throw + minor fix.
* Split LinearConstraint into LinearConstraint and BoxConstraint.
* Add LinearConstraint::constantF
  This implements the Python Constraint.identical attribute.
* Add generation of documentation.
* Add LinearConstraint.
* Merge branch 'feat-cpp' into develop
  Initial setup for toppra-cpp
* Update README.md to include build instruction.
* Remove submodule. Use the officially recommended approach.
* Add googletest as a submodule.
* Add a single empty gtest executable.
* Add some standard headers.
* Contributors: Hung, Hung Pham (Phạm Tiến Hùng), Joseph Mirabel

0.2.3 (2020-01-25)
------------------

0.2.2 (2019-04-27)
------------------

0.2.1 (2019-04-13)
------------------

0.1.1 (2018-05-15)
------------------
