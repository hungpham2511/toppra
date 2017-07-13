from toppra import PathConstraint, interpolate_constraint
from scipy.linalg import block_diag
import numpy as np
import pytest


# C: Canonical; I: TypeI; II: TypeII
intp_data = [
    (2, 0, 0, 0), (5, 0, 0, 0), (0, 3, 5, 0),     # I
    (0, 3, 5, 2),                                 # II
    (0, 3, 5, 2), (0, 4, 5, 0), (3, 0, 0, 0),      # II, I, C
    (0, 3, 1, 2), (0, 4, 1, 0), (3, 0, 0, 0)      # II, I, C
][:]


@pytest.fixture(params=intp_data)
def intp_fixture(request):
    """
    """
    N = 20
    ss = np.linspace(0, 1, N+1)
    nm, neq, nv, niq = request.param
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

    pc = PathConstraint(a=a, b=b, c=c,
                        abar=abar, bbar=bbar, cbar=cbar, D=D,
                        l=l, h=h,
                        lG=lG, G=G, hG=hG, ss=ss,
                        name="RandomPC")
    pc_intp = interpolate_constraint(pc)
    return pc, pc_intp


class TestFunc_interpolate_constraint(object):
    """ Test suite for

    function:
    ---------
       interpolate_constraint

    """
    def test_shape(self, intp_fixture):
        """ Shape check
        """
        pc, pc_intp = intp_fixture

        assert pc_intp.nm == 2 * pc.nm
        assert pc_intp.niq == 2 * pc.niq
        assert pc_intp.neq == 2 * pc.neq
        assert pc_intp.nv == 2 * pc.nv
        assert pc_intp.kind == pc.kind
        assert np.allclose(pc_intp.ss, pc.ss)

    def test_canonical_mat(self, intp_fixture):
        """
        """
        pc, pc_intp = intp_fixture
        # number
        for i in range(pc_intp.N):
            ds = pc_intp.ss[i+1] - pc_intp.ss[i]
            ai_new = np.hstack((
                pc.a[i],
                pc.a[i+1] + 2 * ds * pc.b[i+1]))
            bi_new = np.hstack((pc.b[i], pc.b[i+1]))
            ci_new = np.hstack((pc.c[i], pc.c[i+1]))

            assert np.allclose(ai_new, pc_intp.a[i])
            assert np.allclose(bi_new, pc_intp.b[i])
            assert np.allclose(ci_new, pc_intp.c[i])

    def test_equality_mat(self, intp_fixture):
        """ Equality constraint: abar, bbar, cbar, D
        """
        pc, pc_intp = intp_fixture
        # number
        for i in range(pc_intp.N):
            ds = pc_intp.ss[i+1] - pc_intp.ss[i]
            ai_new = np.hstack((
                pc.abar[i],
                pc.abar[i+1] + 2 * ds * pc.bbar[i+1]))
            bi_new = np.hstack((pc.bbar[i], pc.bbar[i+1]))
            ci_new = np.hstack((pc.cbar[i], pc.cbar[i+1]))
            Di_new = block_diag(pc.D[i], pc.D[i+1])

            li_new = np.hstack((pc.l[i], pc.l[i+1]))
            hi_new = np.hstack((pc.h[i], pc.h[i+1]))

            assert np.allclose(ai_new, pc_intp.abar[i])
            assert np.allclose(bi_new, pc_intp.bbar[i])
            assert np.allclose(ci_new, pc_intp.cbar[i])
            assert np.allclose(Di_new, pc_intp.D[i], atol=1e-8)

            assert np.allclose(li_new, pc_intp.l[i])
            assert np.allclose(hi_new, pc_intp.h[i])

    def test_inequality_mat(self, intp_fixture):
        """ Inequality:
        """
        pc, pc_intp = intp_fixture
        # number
        for i in range(pc_intp.N):
            lGi_new = np.hstack((pc.lG[i], pc.lG[i+1]))
            hGi_new = np.hstack((pc.hG[i], pc.hG[i+1]))

            assert np.allclose(lGi_new, pc_intp.lG[i])
            assert np.allclose(hGi_new, pc_intp.hG[i])
