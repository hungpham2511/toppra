#ifndef TOPPRA_SOLVER_HPP
#define TOPPRA_SOLVER_HPP

#include <toppra/toppra.hpp>

namespace toppra {

/** \brief The base class for all solver wrappers.
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
    /// \brief Create a solver based on the compilation option.
    /// At the time of writing, the preference order is
    /// - qpOASES
    /// - GLPK
    /// If none of these is available, this function returns a null pointer.
    static SolverPtr createDefault();

    /// \copydoc Solver::m_deltas
    const Vector& deltas () const
    {
      return m_deltas;
    }

    /// \copydoc Solver::m_N
    std::size_t nbStages () const
    {
      return m_N;
    }

    /// \copydoc Solver::m_nV
    std::size_t nbVars () const
    {
      return m_nV;
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

    virtual bool solveStagewiseBatch(int i, const Vector& g){};

    virtual bool solveStagewiseBack(int i, const Vector& g, const Bound& xNext, Vector& solution){};

    /// \brief Initialize the solver
    /// \note Child classes should call the parent implementation.
    virtual void initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    /** \brief Initialize the wrapped solver
     */
    virtual void setupSolver ()
    {}

    /** \brief Free the wrapped solver
     */
    virtual void closeSolver ()
    {}

    virtual ~Solver () {}

  protected:
    Solver () {}

    void init (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times);

    struct LinearConstraintParams {
      int cid;
      Vectors a, b, c, g;
      Matrices F;
    };
    struct BoxConstraintParams {
      int cid;
      Bounds u, x;
    };
    struct ConstraintsParams {
      std::vector<LinearConstraintParams> lin;
      std::vector<BoxConstraintParams   > box;
    } m_constraintsParams;

    LinearConstraintPtrs m_constraints;
    GeometricPathPtr m_path;
    Vector m_times;

  private:
    /// \brief Number of stages.
    /// The number of gridpoints equals N + 1, where N is the number of stages.
    std::size_t m_N;
    /// Total number of variables, including u, x.
    std::size_t m_nV;
    /// Time increment between each stage. Size \ref nbStages
    Vector m_deltas;

}; // class Solver

} // namespace toppra

#endif
