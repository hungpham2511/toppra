#include <toppra/solver/glpk-wrapper.hpp>

#ifdef BUILD_WITH_GLPK
# include <glpk.h>
#endif

#include <iostream>

namespace toppra {
namespace solver {

#ifdef BUILD_WITH_GLPK
void intersection (const Bound& a, const Bound& b, Bound& c)
{
  c[0] = std::max(a[0], b[0]);
  c[1] = std::min(a[1], b[1]);
}

int bnd_type(const value_type& l, const value_type& u)
{
  if (l == -infty) {
    if (u == infty) return GLP_FR;
    else return GLP_UP;
  } else {
    if (u < l) {
      TOPPRA_LOG_DEBUG("GLPK: invalid bounds:"
          "\nlower: " << l <<
          "\nupper: " << u);
      // Do not stop here so that all the inequalities are checked.
      // GLPK will report failure.
    }
    if (l == u) return GLP_FX;
    else if (u == infty) return GLP_LO;
    else return GLP_DB;
  }
}

void set_col_bnds(glp_prob* lp, int i, const Bound& ub)
{
  glp_set_col_bnds(lp, i, bnd_type(ub[0], ub[1]), ub[0], ub[1]);
}

void set_row_bnds(glp_prob* lp, int i, const value_type& l, const value_type& u)
{
  glp_set_row_bnds(lp, i, bnd_type(l,u), l, u);
}
void set_row_bnds(glp_prob* lp, int i, const Bound& b) { set_row_bnds(lp, i, b[0], b[1]); }
#endif

void GLPKWrapper::initialize (const LinearConstraintPtrs& constraints, const GeometricPathPtr& path,
        const Vector& times)
{
#ifdef BUILD_WITH_GLPK
  Solver::initialize (constraints, path, times);
  if (m_lp != NULL) glp_delete_prob(m_lp);
  m_lp = glp_create_prob();

  // Currently only support Canonical Linear Constraint
  assert(nbVars() == 2);

  glp_set_obj_dir(m_lp, GLP_MIN);

  // auxiliary variables
  glp_add_cols(m_lp, 2);
  glp_set_col_name(m_lp, 1, "u");
  glp_set_col_name(m_lp, 2, "x");
  glp_set_col_bnds(m_lp, 1, GLP_FR, 0., 0.);

  // structural variables
  int nC = 1;
  for (const Solver::LinearConstraintParams& linParam : m_constraintsParams.lin)
    nC += linParam.F[0].rows();

  glp_add_rows(m_lp, nC);
  glp_set_row_name(m_lp, 1, "x_next");
#else
  TOPPRA_UNUSED(constraints);
  TOPPRA_UNUSED(path);
  TOPPRA_UNUSED(times);
  throw std::invalid_argument("TOPPRA was not built with GLPK support.");
#endif
}

GLPKWrapper::~GLPKWrapper ()
{
#ifdef BUILD_WITH_GLPK
  if (m_lp != NULL) glp_delete_prob(m_lp);
#endif
}

bool GLPKWrapper::solveStagewiseOptim(std::size_t i,
        const Matrix& H, const Vector& g,
        const Bound& x, const Bound& xNext,
        Vector& solution)
{
#ifdef BUILD_WITH_GLPK
  if (H.size() > 0)
    throw std::invalid_argument("GLPK can only solve LPs.");

  int N (nbStages());
  assert (i <= N);

  const int ind[] = {0, 1, 2};
  double val[] = {0., 0., 1.};

  if (i < N) {
    // x_next row
    val[1] = 2*deltas()[i];
    glp_set_mat_row(m_lp, 1, 2, ind, val);
    set_row_bnds(m_lp, 1, xNext);
  } else {
    glp_set_mat_row(m_lp, 1, 0, NULL, NULL);
    glp_set_row_bnds(m_lp, 1, GLP_FR, 0., 0.);
  }

  int iC = 2;
  for (const Solver::LinearConstraintParams& lin : m_constraintsParams.lin)
  {
    std::size_t j (lin.F.size() == 1 ? 0 : i);
    const Matrix& _F (lin.F[j]);
    const Vector& _g (lin.g[j]);
    int nC (_F.rows());

    for (int k = 0; k < _F.rows(); ++k) {
      val[1] = _F.row(k) * lin.a[i];
      val[2] = _F.row(k) * lin.b[i];
      glp_set_mat_row(m_lp, iC, 2, ind, val);
      set_row_bnds(m_lp, iC, -infty, _g[k] - _F.row(k) * lin.c[i]);
      ++iC;
    }
  }

  // Bounds on x and u
  Bound xb(x), ub;
  ub << -infty, infty;

  for (const Solver::BoxConstraintParams& box : m_constraintsParams.box)
  {
    if (!box.u.empty()) intersection(ub, box.u[i], ub);
    if (!box.x.empty()) intersection(xb, box.x[i], xb);
  }

  set_col_bnds(m_lp, 1, ub);
  set_col_bnds(m_lp, 2, xb);

  // Fill cost
  glp_set_obj_coef(m_lp, 1, g[0]);
  glp_set_obj_coef(m_lp, 2, g[1]);

  // TODO give a try to glp_interior
  glp_smcp parm;
  glp_init_smcp(&parm);
  parm.msg_lev = GLP_MSG_ERR;
  int ret = glp_simplex(m_lp, &parm);
  if (ret == 0) {
    int status = glp_get_status(m_lp);
    switch (status) {
      case GLP_OPT:
        break;
      case GLP_FEAS:
        TOPPRA_LOG_DEBUG("GLPK: solution is feasible");
        break;
      case GLP_INFEAS:
        TOPPRA_LOG_DEBUG("GLPK: solution is infeasible");
        break;
      case GLP_NOFEAS:
        TOPPRA_LOG_DEBUG("GLPK: problem has no feasible solution");
        return false;
      case GLP_UNBND:
        TOPPRA_LOG_DEBUG("GLPK: problem has unbounded solution");
        break;
      case GLP_UNDEF:
        TOPPRA_LOG_DEBUG("GLPK: solution is undefined");
        return false;
    }
    solution.resize(2);
    solution << glp_get_col_prim(m_lp, 1), glp_get_col_prim(m_lp, 2);
    return true;
  }
  TOPPRA_LOG_DEBUG("GLPK failed. glp_simplex return value: " << ret);
  return false;
#else
  TOPPRA_UNUSED(i);
  TOPPRA_UNUSED(H);
  TOPPRA_UNUSED(g);
  TOPPRA_UNUSED(x);
  TOPPRA_UNUSED(xNext);
  TOPPRA_UNUSED(solution);
  throw std::invalid_argument("TOPPRA was not built with GLPK support.");
#endif
}

} // namespace solver
} // namespace toppra
