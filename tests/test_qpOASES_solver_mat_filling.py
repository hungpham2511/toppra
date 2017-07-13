import pytest
import numpy as np
from scipy.linalg import block_diag
from topp.fastTOPP import (PathConstraint,
                           qpOASESPPSolver,
                           INFTY)


def random_fill(arrays):
    for array in arrays:
        array[:] = np.random.randn(*array.shape)


# C: Canonical; I: TypeI; II: TypeII
qpOASES_solver_mat_params = [
    ((2, 0, 0, 0), ),                                 # C
    ((5, 0, 0, 0), (3, 0, 0, 0)),                   # C, C
    ((2, 0, 0, 0), (5, 0, 0, 0), (3, 0, 0, 0)),     # C, C, C
    ((0, 3, 5, 0), ),                                 # I
    ((0, 3, 5, 0), (3, 0, 0, 0)),                   # I, C
    ((0, 3, 5, 0), (0, 4, 5, 0), (3, 0, 0, 0)),     # I, I, C
    ((0, 3, 5, 2), ),                                 # II
    ((0, 3, 5, 3), (3, 0, 0, 0)),                   # II, C
    ((0, 3, 5, 2), (0, 4, 5, 0), (3, 0, 0, 0))      # II, I, C
][:]
qpOASES_mat_ids = [
    "C", "C, C", "C, C, C",
    "I", "I, C", "I, I, C",
    "II", "II, C", "II, I, C"
][:]


@pytest.fixture(params=qpOASES_solver_mat_params, ids=qpOASES_mat_ids)
def qpOASES_mat_fixtures(request):
    """
    """
    N = 20
    ss = np.linspace(0, 1, N+1)
    pcs = []
    for param in request.param:
        nm, neq, nv, niq = param
        a = np.random.randn(N+1, nm)
        b = np.random.randn(N+1, nm)
        c = np.random.randn(N+1, nm)

        abar = np.random.randn(N+1, neq)
        bbar = np.random.randn(N+1, neq)
        cbar = np.random.randn(N+1, neq)

        D = np.random.randn(N+1, neq, nv)
        l = np.random.randn(N+1, nv)
        h = np.random.randn(N+1, nv)

        G = np.random.randn(N+1, niq, nv)
        lG = np.random.randn(N+1, niq)
        hG = np.random.randn(N+1, niq)

        pcs.append(PathConstraint(a=a, b=b, c=c,
                                  abar=abar, bbar=bbar, cbar=cbar, D=D,
                                  l=l, h=h,
                                  lG=lG, G=G, hG=hG, ss=ss,
                                  name="RandomPC"))

    return pcs, qpOASESPPSolver(pcs)


class TestClass_qpOASES_solver_matrices(object):
    """After initialization, the solver produce temporary matrices
    A, lA, hA, l, h satisfy following condition

        lA[i] <= A[i] (u, x, v') <= hA[i]
        l[i]  <=      (u, x, v') <= h[i]

    Afterward at each time step it re_fill all matrices according to
    the given index. For example, if `_re_fill` with index i = 0, then
    all constraint matrices are filled with coefficient at i =0.
    """

    def test_shape_after_init(self, qpOASES_mat_fixtures):
        """ Verify dimension.
        """
        pcs, pp = qpOASES_mat_fixtures

        assert pp.nv == sum([pc.nv for pc in pcs])
        assert pp.niq == sum([pc.niq for pc in pcs])
        assert pp.neq == sum([pc.neq for pc in pcs])
        assert pp.nm == sum([pc.nm for pc in pcs])

        assert pp.nV == (pp.nv + 2)  # qpOASES variable (u, x, v)
        assert pp.nC == (pp.niq + pp.neq + pp.nm + pp.nop)
        # qpOASES constraint

        assert np.allclose(pp.A.shape, (pp.N+1, pp.nC, pp.nV))
        assert (pp.A.shape[1] == pp.lA.shape[1] and
                pp.A.shape[1] == pp.hA.shape[1])
        assert pp.A.shape[2] == pp.nV
        assert pp.g.shape[0] == pp.nV
        assert pp.H.shape == (pp.nV, pp.nV)

    def test_matrices_H_g_after_func_fill(self, qpOASES_mat_fixtures):
        """ Verify qpOASES matrices after filling.
        """
        pcs, pp = qpOASES_mat_fixtures

        # random alteration
        random_fill([pp.H, pp.g])
        # fill
        pp._fill_matrices()
        assert np.allclose(pp.H, np.zeros((pp.nV, pp.nV)))
        assert np.allclose(pp.g, np.zeros(pp.nV))

    def test_matrices_A_after_func_fill(self, qpOASES_mat_fixtures):
        """ Verify qpOASES matrices after filling.
        """
        pcs, pp = qpOASES_mat_fixtures

        random_fill([pp.A])
        pp._fill_matrices()
        # A
        # operational rows
        for i in range(pp.N+1):
            assert np.allclose(pp.A[i, :pp.nop, :], np.zeros((pp.nop, pp.nV)))

            # Assert canonical part if there are canonical constraints
            a_expected = np.hstack(map(lambda pc: pc.a[i], pcs))
            b_expected = np.hstack(map(lambda pc: pc.b[i], pcs))

            assert np.allclose(pp.A[i, pp.nop: pp.nop + pp.nm, 0], a_expected)
            assert np.allclose(pp.A[i, pp.nop: pp.nop + pp.nm, 1], b_expected)
            assert np.allclose(pp.A[i, pp.nop: pp.nop + pp.nm, 2:],
                               np.zeros((pp.nm, pp.nv)))

            a_expected = np.hstack(map(lambda pc: pc.abar[i], pcs))
            assert np.allclose(
                pp.A[i, pp.nop + pp.nm: pp.nop + pp.nm + pp.neq, 0],
                a_expected)

            b_expected = np.hstack(map(lambda pc: pc.bbar[i], pcs))
            assert np.allclose(
                pp.A[i, pp.nop + pp.nm: pp.nop + pp.nm + pp.neq, 1],
                b_expected)

            D_expected = block_diag(*map(lambda pc: - pc.D[i], pcs))
            assert np.allclose(
                pp.A[i, pp.nop + pp.nm: pp.nop + pp.nm + pp.neq, 2:],
                D_expected)

            G_expected = block_diag(*map(lambda pc: pc.G[i], pcs))
            assert np.allclose(
                pp.A[i, pp.nop + pp.nm + pp.neq:
                     pp.nop + pp.nm + pp.neq + pp.niq, 2:], G_expected)

    def test_matrices_lA_hA_after_func_fill(self, qpOASES_mat_fixtures):
        """ Verify qpOASES matrices after filling.
        """
        pcs, pp = qpOASES_mat_fixtures

        i = 0
        random_fill([pp.lA, pp.hA])
        pp._fill_matrices()
        for i in range(pp.N+1):
            # operational rows
            assert np.allclose(pp.lA[i, :pp.nop], np.zeros(pp.nop))
            assert np.allclose(pp.hA[i, :pp.nop], np.zeros(pp.nop))
            # canonical part
            c_expected = np.hstack(map(lambda pc: pc.c[i], pcs))
            assert np.allclose(pp.lA[i, pp.nop: pp.nop + pp.nm], - INFTY)
            assert np.allclose(pp.hA[i, pp.nop: pp.nop + pp.nm], - c_expected)

            # equality part
            c_expected = np.hstack(map(lambda pc: pc.cbar[i], pcs))
            assert np.allclose(
                pp.lA[i, pp.nop + pp.nm: pp.nop + pp.nm + pp.neq],
                - c_expected)
            assert np.allclose(
                pp.hA[i, pp.nop + pp.nm: pp.nop + pp.nm + pp.neq],
                - c_expected)

            # inequality part
            lG_expected = np.hstack(map(lambda pc: pc.lG[i], pcs))
            assert np.allclose(
                pp.lA[i, pp.nop + pp.nm + pp.neq:], lG_expected)

            hG_expected = np.hstack(map(lambda pc: pc.hG[i], pcs))
            assert np.allclose(
                pp.hA[i, pp.nop + pp.nm + pp.neq:], hG_expected)

    def test_matrices_l_h_after_func_fill(self, qpOASES_mat_fixtures):
        """ Verify qpOASES matrices after filling.
        """
        pcs, pp = qpOASES_mat_fixtures

        random_fill([pp.l, pp.h])
        pp._fill_matrices()
        for i in range(pp.N+1):
            assert pp.l[i, 0] == -INFTY
            assert pp.l[i, 1] == 0
            assert pp.h[i, 0] == INFTY
            assert pp.h[i, 1] == INFTY

            l_expected = np.hstack(map(lambda pc: pc.l[i], pcs))
            h_expected = np.hstack(map(lambda pc: pc.h[i], pcs))
            assert np.allclose(pp.l[i, 2:], l_expected)
            assert np.allclose(pp.h[i, 2:], h_expected)


