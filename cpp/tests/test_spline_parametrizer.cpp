#include <toppra/parametrizer/spline.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include "gtest/gtest.h"

#define TEST_PRECISION 1e-15


class ParametrizeSpline : public testing::Test {
public:
    ParametrizeSpline() {
        toppra::Matrix coeff0{4, 2}, coeff1{4, 2}, coeff2{4, 2};
        coeff0 << -0.500000, -0.500000, 1.500000, 0.500000, 0.000000, 3.000000, 0.000000,
                0.000000;
        coeff1 << -0.500000, -0.500000, 0.000000, -1.000000, 1.500000, 2.500000, 1.000000,
                3.000000;
        coeff2 << -0.500000, -0.500000, -1.500000, -2.500000, 0.000000, -1.000000, 2.000000,
                4.000000;
        toppra::Matrices coefficents = {coeff0, coeff1, coeff2};
        path = std::make_shared<toppra::PiecewisePolyPath>(
                coefficents, std::vector<toppra::value_type>{0, 1, 2, 3});
    };
    std::shared_ptr<toppra::PiecewisePolyPath> path;
};

TEST_F(ParametrizeSpline, BasicUsage) {
    toppra::Vector gridpoints = toppra::Vector::LinSpaced(10, 0, 3);
    toppra::Vector vsquared{10};
    vsquared << 1, 0.9, 0.8, 0.7, 0.6, 0.6, 0.7, 0.8, 0.9, 1;
    auto p = toppra::parametrizer::Spline(path, gridpoints, vsquared);

    // Assert q at each gridpoint
    toppra::value_type t = 0.0;
    toppra::Vector actual_q = p.eval_single(t), desired_q = path->eval_single(t);
    for (size_t i = 1; i < gridpoints.rows(); i++) {
        ASSERT_NEAR(actual_q[0], desired_q[0], TEST_PRECISION);
        ASSERT_NEAR(actual_q[1], desired_q[1], TEST_PRECISION);
        t += (gridpoints[i] - gridpoints[i - 1]) / (std::sqrt(vsquared[i]) + std::sqrt(vsquared[i - 1])) * 2;
        actual_q = p.eval_single(t);
        desired_q = path->eval_single(gridpoints[i]);
    }

    toppra::Bound path_interval = p.pathInterval();
    toppra::Vector ts = toppra::Vector{2};
    ts << path_interval[0], path_interval[1];
    toppra::Vectors qd = p.eval(ts, 1);
    toppra::Vector qd_init (2), qd_final (2);
    qd_init << 0 * std::sqrt(vsquared[0]), 3 * std::sqrt(vsquared[0]);
    qd_final <<
        (-1.5 * std::pow(3 - 2, 2) - 3 * (3 - 2)) * std::sqrt(vsquared[vsquared.rows() - 1]),
        (-1.5 * std::pow(3 - 2, 2) - 5 * (3 - 2) - 1) * std::sqrt(vsquared[vsquared.rows() - 1]);

    // Assert qd at endpoints
    ASSERT_NEAR(qd[0][0], qd_init[0], TEST_PRECISION);
    ASSERT_NEAR(qd[0][1], qd_init[1], TEST_PRECISION);
    ASSERT_NEAR(qd[1][0], qd_final[0], TEST_PRECISION);
    ASSERT_NEAR(qd[1][1], qd_final[1], TEST_PRECISION);
}
