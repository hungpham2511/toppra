#ifndef TOPPRA_SOLVER_HPP
#define TOPPRA_SOLVER_HPP

#include <toppra/toppra.hpp>

namespace toppra {

/** The base class for all solver wrappers.
 *
 *  All Solver can solve Linear/Quadratic Program subject to linear constraints
 *  at the given stage, and possibly with additional auxiliary constraints.
 *
 *  All Solver derived class implements
 *  - Solver::solveStagewiseOptim: core method needed by all Reachability
 *    Analysis-based algorithms
 *  - Solver::setupSolver, Solver::closeSolver: needed by some Solver
 *    implementation, such as mosek and qpOASES with warmstart.
 *
 *  Note that some Solver only handle Linear Program while
 *  some handle both.
 *
 *  Each solver wrapper should provide solver-specific constraint,
 *  such as ultimate bound the variable u, x. For some solvers such as
 *  ECOS, this is very important.
 *
 * */
class Solver {
  public:
    /// \copydoc Solver::deltas_
    const Vector& deltas () const
    {
      return deltas_;
    }

    /// \copydoc Solver::N_
    std::size_t nbStages () const
    {
      return N_;
    }

    /// \copydoc Solver::nV_
    std::size_t nbVars () const
    {
      return nV_;
    }

    /** Solve a stage-wise quadratic (or linear) optimization problem.
     *
     *  The quadratic optimization problem is described below:
     *
     *  \f{eqnarray}
     *      \text{min  }  & 0.5 [u, x, v] H [u, x, v]^\top + [u, x, v] g    \\
     *      \text{s.t.  } & [u, x] \text{ is feasible at stage } i          \\
     *                    & x_{min} \leq x \leq x_{max}                     \\
     *                    & x_{next, min} \leq x + 2 \Delta_i u \leq x_{next, max},
     *  \f}
     *
     *  where `v` is an auxiliary variable, only exist if there are
     *  non-canonical constraints.  The linear program is the
     *  quadratic problem without the quadratic term.
     *
     *  \param i The stage index.
     *  \param H Either a matrix of size (d, d), where d is \ref nbVars, in
     *           which case a quadratic objective is defined, or a matrix
     *           of size (0,0), in which case a linear objective is defined.
     *  \param g Vector of size \ref nbVars. The linear term.
     *  \param[out] solution in case of success, stores the optimal solution.
     *
     *  \return whether the resolution is successful, in which case \c solution
     *          contains the optimal solution.
     *  */
    virtual bool solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution) = 0;

    virtual void setupSolver ()
    {}

    virtual void closeSolver ()
    {}

  protected:
    Solver (const LinearConstraintPtrs& constraints, const GeometricPath& path,
        const Vector& times);

  private:
    LinearConstraintPtrs constraints_;
    const GeometricPath& path_;
    Vector times_;

    /// \brief Number of stages.
    /// The number of gridpoints equals N + 1, where N is the number of stages.
    std::size_t N_;
    /// Total number of variables, including u, x.
    std::size_t nV_;
    /// Time increment between each stage. Size \ref nbStages
    Vector deltas_;

    struct LinearConstraintParams {
      int cid;
      Vectors a, b, c, g;
      Matrices F;
      void init(std::size_t N, LinearConstraint* constraint);
    };
    struct BoxConstraintParams {
      int cid;
      Bounds u, x;
      void init(std::size_t N, BoxConstraint* constraint);
    };
    struct ConstraintsParams {
      std::vector<LinearConstraintParams> lin;
      std::vector<BoxConstraintParams   > box;
    } constraintsParams_;

}; // class Solver
} // namespace toppra

#endif
