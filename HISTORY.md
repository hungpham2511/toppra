# History

## [unrelease]

### Added
- [cpp] Implement serialization/deserialization for piecewise poly trajectory.
- [cpp] Provide Python bindings for PiecewisePolyPath, TOPPRA and constraint classes.
- [cpp] Construct piecewise poly as a hermite spline.
- [python] Provide an option to allow Python 2.7 usage
- [python] Post-processing is now done via parametrizer classes.

### Changed
- [cpp] Minor improvement to PiecewisePolyPath.
- [python] Implement `ParametrizeConstAccel` to allow reparametrize path.
- [python] Parametrization output accessible via ParameterizationData class.
- [python] Remove useless `scaling` in parameter computation.
- [cpp] Clamp velocities to be within controllable sets.


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
[cm-0e022c]: https://github.com/hungpham2511/toppra/commit/0e022c53ab9db473485bd9fb6b8f34a7364efdf8
