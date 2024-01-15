# Changelog

- [cpp]: fix: Seidel LP 1D: incoherent bounds (#244)
- [cpp]: fix: Cannot convert from 'initializer list' to 'toppra::BoundaryCond' (#245)

## 0.6.2 (Sept 19 2023)

### Changed

- [cpp]: warn when solver fails. (#241)

## 0.6.1

### Changed

- [cpp]: fix numerical issue in Seidel solver for 1D problems. (#232)

## 0.6.0 (Mar 08 2023)

### Added
- [python] seidelWrapper solves 1d LP instead of 2d LP when x_min == x_max and solve_1d > 0
- [python] Fixed planning_utils.py to follow the latest api of toppra.algorithm.ParameterizationAlgorithm
- [cpp] add initial gridpoints to proposeGridpoints method.

### Changed
- fix: the f-string bug when hot_qpoases solver failed (#208)
- feat(solver): Make seidel solver(cython) solves 1D LP when possible (#223)
- feat(cpp,constraint): Make Interpolation the default method (#225)
- fix(util): toppra.compute_trajectory() does not return aux data. (#224)

## 0.5.2 (Nov 19 2022)
- [cpp] always define all installed symbols.

## 0.5.0 (July 08 2021)

- [cpp] Minor PR to make the cpp part compiles on windows64 (msvc). Thanks @ahoarau.
- [cpp] Fixed out of bounds issue in parametrizer::ConstAccel.
- [cpp] Fixed incorrect variable bug in seidel solver.
- [cpp] [#187] Expose waypoint times in parametizers.
- [python] Fix linting error that causes CI failure
- [ci] Do integration testing in gh action instead of circle CI
- [ci] Remove Circle CI and use Github Action for all testings

## 0.4.2 (Aug 08 2021)
- [cpp] Replace for loop in piecewise poly path to find index by std::lower_bound.


## 0.4.1 (May 06 2021)

### Added
- [python] Fixed build error for numpy versions <= 1.20
- [cpp] Implement spline parametrizer

## 0.4.0 (Mar 16 2021)

### Added
- [python] Fixed build error when cython and numpy were not pre-installed
- [python] Fix a bug that prevent user from using the ConstAccelParametrizer
- [python] Allow specifying minimum nb of grid-points during automatic selection
- [python] Minor performance tweak in reachability_algorithm.py
- [ci] Fix python2.7 dependencies
- [python] Minor bug fixed in the SimplePath class
- [cpp] [#129][#129] Implement constant-acceleration trajectory parametrizer
- [cpp] [#143][#143] Add PathParametrizationAlgorithm::setGridpoints
- [cpp] [#146][#146] Add constraint::CartesianVelocityNorm
- [cpp] [#158][#158] Add Seidel solver.
- [cpp] Enhance Seidel solver numerical behaviour.
- [cpp] [#161] Enable to construct spline from 1st and 2nd order derivatives at the curve endpoints

### Changed
- [cpp] [#153] Fix variable mismatch in constraint::CartesianVelocityNorm

### Improved Documentation

- Switch to furo theme for better readability.

## 0.3.1 (Aug 23 2020)

### Added
- [docs] Use example gallery to show examples.
- [cpp] Implement serialization/deserialization for piecewise poly trajectory.
- [cpp] Provide Python bindings for PiecewisePolyPath, TOPPRA and constraint classes.
- [cpp] Construct piecewise poly as a hermite spline.
- [cpp] Add varying joint velocity limits.
- [python] [#117] Provide an option to allow Python 2.7 usage
- [python] [#117] Post-processing is now done via parametrizer classes.

### Changed
- [python]Add some type annotations to parameterizer.py
- [python]Support older interpolation method.
- [cpp] Minor improvement to PiecewisePolyPath.
- [python] Implement `ParametrizeConstAccel` to allow reparametrize path.
- [python] Parametrization output accessible via ParameterizationData class.
- [python] Remove useless `scaling` in parameter computation.
- [cpp] Clamp velocities to be within controllable sets.
- [ci] [#117] Improve CI pipeline to test on several python versions
- [ci] [#139] Automate publish to PyPI server


## 0.3.0 (May 3 2020)

Major release! Implement TOPPRA in C++ and several improvements to Python codebase.

### Added

- [cpp] Add solver wrapper based on GLPK.
- [cpp] Initial cpp TOPPRA implementation: compute parametrization, feasible sets, qpOASES solver wrapper.
- [python] Implement a new trajectory class for specified velocities.

### Changed

- [python] Improve documentation for `toppra.constraint`
- [python] #98: Eliminate use of deprecated method.
- [cpp] Bug fixes in solver wrapper.
- [python] Simplify TOPPRA class interface.
- [python] [0e022c][cm-0e022c] Update README.md to reflect development roadmap.
- [python] Format some source files with black.
- [python] [#78][gh-78] Improve documentation structure.
- [python] [#79][gh-79] Improve documentation structure.

### Removed

- Dropping support for Python2.7. Adding type annotation compatible to Python 3.

## 0.2.3 (Jan 25 2020)

- Auto-gridpoints feature (#73)

## <0.2.3
- Many commits and features.

[gh-78]: https://github.com/hungpham2511/toppra/pull/78
[gh-79]: https://github.com/hungpham2511/toppra/pull/79
[#117]: https://github.com/hungpham2511/toppra/pull/117
[#129]: https://github.com/hungpham2511/toppra/pull/129
[#143]: https://github.com/hungpham2511/toppra/pull/143
[#146]: https://github.com/hungpham2511/toppra/pull/146
[#158]: https://github.com/hungpham2511/toppra/pull/158
[cm-0e022c]: https://github.com/hungpham2511/toppra/commit/0e022c53ab9db473485bd9fb6b8f34a7364efdf8
