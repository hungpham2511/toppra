#include <toppra/parametrizer/spline.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>

namespace toppra {

namespace parametrizer {

Spline::Spline(GeometricPathPtr path, const Vector& gridpoints, const Vector& vsquared)
    : Parametrizer(path, gridpoints, vsquared) {
        std::vector<value_type> t_grid (gridpoints.rows(), 0);
        std::vector<value_type> copied_gridpoints (gridpoints.data(), gridpoints.data() + gridpoints.rows());
        std::vector<int> skip_ent;
        value_type sd_average, delta_s, delta_t;
        for (int i = 1; i < t_grid.size(); i++) {
            sd_average = (m_vs[i - 1] + m_vs[i]) / 2;
            delta_s = gridpoints[i] - gridpoints[i - 1];
            if (sd_average > TOPPRA_NEARLY_ZERO) {
                delta_t = delta_s / sd_average;
            }
            else {
                delta_t = 5;
            }
            t_grid[i] = t_grid[i - 1] + delta_t;
            if (delta_t < TOPPRA_NEARLY_ZERO) {
                skip_ent.push_back(i);
            }
        }

        for (int i = skip_ent.size() - 1; i > -1; i--) {
            t_grid.erase(t_grid.begin() + skip_ent[i]);
            copied_gridpoints.erase(copied_gridpoints.begin() + skip_ent[i]);
        }

        Vectors q_grid = path->eval(Eigen::Map<Vector>(copied_gridpoints.data(), copied_gridpoints.size()));
        m_ts = Eigen::Map<Vector>(t_grid.data(), t_grid.size());

        Bound path_interval = m_path->pathInterval();
        BoundaryCond init_bc, final_bc;
        init_bc.order = final_bc.order = 1;
        init_bc.values = m_path->eval_single(path_interval[0], 1) * m_vs[0];
        final_bc.values = m_path->eval_single(path_interval[1], 1) * m_vs[m_vs.rows() - 1];
        m_path = std::make_shared<PiecewisePolyPath>(q_grid, m_ts, std::array<BoundaryCond, 2>{init_bc, final_bc});
}

Vectors Spline::eval_impl(const Vector &times, int order) const {
    return m_path->eval(times, order);
}

bool Spline::validate_impl() const {
    return true;
}

Bound Spline::pathInterval_impl() const {
    Bound b;
    b << m_ts[0], m_ts[m_ts.size() - 1];
    return b;
}

}  // namespace parametrizer

}  // namespace toppra
