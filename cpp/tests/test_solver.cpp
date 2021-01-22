#ifdef BUILD_WITH_qpOASES
#include <toppra/solver/qpOASES-wrapper.hpp>
#endif
#ifdef BUILD_WITH_GLPK
#include <toppra/solver/glpk-wrapper.hpp>
#endif
#include <toppra/solver/seidel.hpp>

#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>

#include "gtest/gtest.h"

constexpr int Ntrials = 10;

std::map<std::string, toppra::Vectors> solutions;

class Solver : public testing::Test {
public:
  Solver() : path(constructPath()), g(Eigen::Vector2d{0.5, 2.}) {

    // path has the equation: 0 * x ^ 3 + 1 * x ^ 2 + 2 x ^ 1 + 3
  }

  std::shared_ptr<toppra::PiecewisePolyPath> constructPath() {
    toppra::Matrix coeff{4, 2};
    coeff.colwise() = toppra::Vector::LinSpaced(4, 0, 3);
    toppra::Matrices coefficents = {coeff, coeff};
    return std::make_shared<toppra::PiecewisePolyPath>(
        coefficents, std::vector<double>{0, 1, 2});
  }

  toppra::Vector getTimes (int N)
  {
    toppra::Bound I (path->pathInterval());
    return toppra::Vector::LinSpaced (N, I[0], I[1]);
  }

  std::shared_ptr<toppra::PiecewisePolyPath> path;
  toppra::Vector g;

  void testSolver(toppra::Solver& solver, const char* name)
  {
    using namespace toppra;

    int nDof = path->dof();
    Vector lb (-Vector::Ones(nDof)),
           ub ( Vector::Ones(nDof));
    LinearConstraintPtr ljv (new constraint::LinearJointVelocity (lb, ub));
    LinearConstraintPtr lja (new constraint::LinearJointAcceleration (lb, ub));

    int N = Ntrials;
    Vector times (getTimes(N));

    solver.initialize ({ ljv, lja }, path, times);

    EXPECT_EQ(solver.nbStages(), N-1);
    EXPECT_EQ(solver.nbVars(), 2);
    ASSERT_EQ(solver.deltas().size(), N-1);
    for (int i = 0; i < N-1; ++i)
      EXPECT_NEAR(solver.deltas()[i], times[i+1] - times[i], 1e-10);

    solver.setupSolver();
    Vector solution;
    Matrix H;
    const value_type infty (std::numeric_limits<value_type>::infinity());
    Bound x, xNext;
    x << -infty, infty;
    xNext << -infty, infty;
    Vectors sols;
    for (int i = 0; i < N; ++i) {
      EXPECT_TRUE(solver.solveStagewiseOptim(i, H, g, x, xNext, solution));
      EXPECT_EQ(solution.size(), solver.nbVars());
      sols.emplace_back(solution);
    }
    solver.closeSolver();

    solutions.emplace(name, sols);
  }
};

#ifdef BUILD_WITH_qpOASES
TEST_F(Solver, qpOASESWrapper) {
  toppra::solver::qpOASESWrapper solver;
  testSolver(solver, "qpOASES");
}
#endif

#ifdef BUILD_WITH_GLPK
TEST_F(Solver, GLPKWrapper) {
  toppra::solver::GLPKWrapper solver;
  testSolver(solver, "GLPK");
}
#endif

TEST_F(Solver, Seidel) {
  toppra::solver::Seidel solver;
  testSolver(solver, "Seidel");
}

// Check that all the solvers returns the same solution.
// TODO each solver is expected to be tested on the same inputs. It should be
// templated, so that we know the same problem is setup (with a template hook to
// enable adding code specific to one solver).
TEST_F(Solver, consistency)
{
  auto ref = solutions.begin();
  bool first = true;
  for(const auto& pair : solutions) {
    if (first) {
      first = false;
      continue;
    }
    ASSERT_EQ(pair.second.size(), ref->second.size());
    for (int i = 0; i < pair.second.size(); ++i) {
      ASSERT_EQ(pair.second[i].size(), ref->second[i].size());
      for (int j = 0; j < pair.second[i].size(); ++j) {
        EXPECT_NEAR(pair.second[i][j], ref->second[i][j], 1e-6)
          << " solvers " << ref->first << " and " << pair.first << " disagree.";
      }
    }
  }
}

#include <toppra/solver/seidel-internal.hpp>

TEST(SeidelFunctions, seidel_1d) {
  using namespace toppra::solver;

  {
    /*
     * max   x
     * s.t.  x - 1e11 <= 0
     */
    RowVector2 v (1, 0);
    MatrixX2 A (1, 2);
    A << 1, -seidel::INF*10;
    auto sol = seidel::solve_lp1d(v, A);
    EXPECT_TRUE(sol.feasible);
    EXPECT_DOUBLE_EQ(-A(0,1), sol.optvar[0]);
  }

  {
    /*
     * max   -x
     * s.t.   x - 1e11 <= 0
     */
    RowVector2 v (-1, 0);
    MatrixX2 A (1, 2);
    A << 1, -seidel::INF*10;
    auto sol = seidel::solve_lp1d(v, A);
    EXPECT_TRUE(sol.feasible);
    EXPECT_DOUBLE_EQ(-seidel::infinity, sol.optvar[0]);
  }

  {
    /*
     * max   -x
     * s.t.  -x + 3 <= 0
     */
    RowVector2 v (-1, 0);
    MatrixX2 A (1, 2);
    A << -1, 3;
    auto sol = seidel::solve_lp1d(v, A);
    EXPECT_TRUE(sol.feasible);
    EXPECT_DOUBLE_EQ(3, sol.optvar[0]);
  }
}

TEST(SeidelFunctions, seidel_2d) {
  using namespace toppra::solver;
  RowVector2 v;
  MatrixX3 A;
  Vector2 low, high;
  std::array<int, 2> active_c;
  bool use_cache = false;
  std::vector<int> index_map;
  MatrixX2 A_1d;
  seidel::LpSol sol;

  using Eigen::VectorXd;
  auto LooseNegative = [](const Eigen::VectorXd &a) { return (a.array() < seidel::TINY).all(); };

  auto check = [&LooseNegative, &A, &low, &high, &sol](bool expectFeasible){
    EXPECT_EQ(expectFeasible, sol.feasible);
    if (sol.feasible) {
      Eigen::VectorXd A_times_optvar = A.leftCols<2>() * sol.optvar + A.col(2);

      // Check that all constraints are statisfied.
      EXPECT_PRED1(LooseNegative, A_times_optvar);
      // Check that active constraints are approximatevely zero.
      for (int i = 0; i < 2; ++i) {
        ASSERT_LT(sol.active_c[i], A.rows());
        ASSERT_GE(sol.active_c[i], -4);
        if (sol.active_c[i] < 0) {
          switch (sol.active_c[i]) {
            case seidel::LOW_0 : EXPECT_NEAR(low [0], sol.optvar[0], seidel::TINY); break;
            case seidel::HIGH_0: EXPECT_NEAR(high[0], sol.optvar[0], seidel::TINY); break;
            case seidel::LOW_1 : EXPECT_NEAR(low [1], sol.optvar[1], seidel::TINY); break;
            case seidel::HIGH_1: EXPECT_NEAR(high[1], sol.optvar[1], seidel::TINY); break;
          }
        } else {
          EXPECT_DOUBLE_EQ(A_times_optvar[sol.active_c[i]], 0.);
        }
      }
    }
  };

  {
    A.resize(16, 3);
    A_1d.resize(A.rows()+4, 2);
    v << 0.04, 1;
    A <<
           -0.04,           -1,           0,
            0.04,            1,    -2.26492,
       -0.087295,    0.0654839,    -28.3501,
        0.258242,    -0.114201,    -43.4429,
      -0.0964134,     0.262025,    -27.1247,
        0.117863,    -0.191702,    -27.7368,
      0.00258571,   0.00680004,    -14.5918,
        0.017961,   -0.0688431,    -14.1562,
    -0.000553256,   0.00212078,    -14.7322,
        0.087295,   -0.0654839,    -28.3499,
       -0.258242,     0.114201,    -13.2571,
       0.0964134,    -0.262025,    -29.5753,
       -0.117863,     0.191702,    -28.9632,
     -0.00258571,  -0.00680004,    -14.8082,
       -0.017961,    0.0688431,    -15.2438,
     0.000553256,  -0.00212078,    -14.6678;

    low << -1e+08, 2.06944;
    high << 1e+08, 2.06944 - 2.22045e-14;

    sol = seidel::solve_lp2d(v, A, low, high,
        active_c, use_cache, index_map, A_1d);

    check(true);
  }
}
