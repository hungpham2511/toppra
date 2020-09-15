from .solverwrapper import SolverWrapper, check_solver_availability
from ..constraint import ConstraintType
from ..constants import INFTY, ECOS_MAXX, ECOS_INFTY
import logging
import numpy as np
import scipy.sparse

try:
    import ecos
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ecosWrapper(SolverWrapper):
    """A solver wrapper that handles linear and conic-quadratic constraints using ECOS.

    :class:`ecosWrapper` and :class:`cvxpyWrapper` are the only
    wrappers that can handle conic-quadratic constraints, which are
    necessary to compute robust path parameterization.

    Notes
    -----
    To reduce numerical-related issues, ECOS_MAXX is used to regulate
    the magnitude of the solution.

    ECOS is not very well implemented. There are many cases in which
    the solver fails simply because there is a very large bound
    (>1e6). Because of this, the test suites included in toppra do not
    include many tests for ecos.


    Attributes
    ----------
    constraints : list of `Constraint`
        Constraints on the robot system.
    path : Interpolator
        The geometric path to be time-parametrized.
    path_discretization: array
        The discretization grid use to discretize the geometric path.

    """

    def __init__(self, constraint_list, path, path_discretization):
        super(ecosWrapper, self).__init__(constraint_list, path, path_discretization)
        if not check_solver_availability('ecos'):
            raise RuntimeError(
                "Unable to start ecos wrapper because ECOS solver is not installed."
            )
        # NOTE: Currently receive params in dense form.
        self._linear_idx = []
        self._conic_idx = []
        # Linear_dim + conic_dim are the number of rows of matrix G and h
        # conic_dim is divisible by 4
        self._linear_dim = 0
        self._conic_dim = 0
        logger.debug("Registering constraints")
        for i, constraint in enumerate(constraint_list):
            logger.debug("Constraint [{:d}] \n {:}".format(i, constraint.__repr__()))
            _type = constraint.get_constraint_type()
            if _type == ConstraintType.CanonicalLinear:
                self._linear_idx.append(i)
                if self.params[i][3] is not None:  # Check F is not None
                    if constraint.identical:
                        self._linear_dim += self.params[i][3].shape[0]
                    else:
                        self._linear_dim += self.params[i][3].shape[1]
                if self.params[i][5] is not None:  # Check ubound
                    self._linear_dim += 2
                if self.params[i][6] is not None:  # Check xbound
                    self._linear_dim += 2
            elif _type == ConstraintType.CanonicalConic:
                self._conic_idx.append(i)
                self._conic_dim += 4 * self.params[i][0].shape[1]
            else:
                raise NotImplementedError(
                    "Constraint type {:} not implemented".format(_type)
                )
        assert self._conic_dim % 4 == 0
        logger.debug("Indices of linear constraints: {:}".format(self._linear_idx))
        logger.debug("Indices of conic constraints : {:}".format(self._conic_idx))
        logger.debug(
            "Nb of row for linear constraints: {:d} rows".format(self._linear_dim)
        )
        logger.debug(
            "Nb of row for conic constraints : {:d} rows".format(self._conic_dim)
        )

    def solve_stagewise_optim(self, i, H, g, x_min, x_max, x_next_min, x_next_max):
        assert i <= self.N and 0 <= i
        assert H is None or np.allclose(H, np.zeros(2))

        # Total number of rows in matrix G, h. Descriptions of terms
        # 1) x_min <= x <= x_max
        # 2) xnext_min <= x + 2 ds u <= xnext_max
        # 3) linear constraints
        # 4) conic constraints
        if i < self.N:
            nrow = 2 + 2 + self._linear_dim + self._conic_dim
            dims = {"l": 2 + 2 + self._linear_dim, "q": [4] * (self._conic_dim // 4)}
        else:
            nrow = 2 + self._linear_dim + self._conic_dim
            dims = {"l": 2 + self._linear_dim, "q": [4] * (self._conic_dim // 4)}

        G_lil = scipy.sparse.lil_matrix((nrow, 2))
        h = np.zeros(nrow)
        # Fill G and h
        currow = 0
        ## Fill 1)
        G_lil[currow : currow + 2, 1] = [[-1], [1]]
        if not np.isnan(x_min):
            h[currow] = -x_min
        else:
            h[currow] = ECOS_INFTY
        currow += 1
        if not np.isnan(x_max):
            h[currow] = x_max
        else:
            h[currow] = ECOS_INFTY
        currow += 1
        ## Fill 2)
        if i < self.N:
            delta = self.get_deltas()[i]
            G_lil[currow, :] = [[-2 * delta, -1]]
            if not np.isnan(x_next_min):
                h[currow] = -x_next_min
            else:
                h[currow] = ECOS_INFTY
            currow += 1
            G_lil[currow, :] = [[2 * delta, 1]]
            if not np.isnan(x_next_max):
                h[currow] = x_next_max
            else:
                h[currow] = ECOS_INFTY
            currow += 1
        ## Fill 3)
        for k in self._linear_idx:
            _a, _b, _c, _F, _h, _ubound, _xbound = self.params[k]

            if _a is not None:
                if self.constraints[k].identical:
                    nb_cnst = _F.shape[0]
                    G_lil[currow : currow + nb_cnst, 0] = np.dot(_F, _a[i]).reshape(
                        -1, 1
                    )
                    G_lil[currow : currow + nb_cnst, 1] = np.dot(_F, _b[i]).reshape(
                        -1, 1
                    )
                    h[currow : currow + nb_cnst] = _h - np.dot(_F, _c[i])
                    currow += nb_cnst
                else:
                    nb_cnst = _F.shape[1]
                    G_lil[currow : currow + nb_cnst, 0] = np.dot(_F[i], _a[i]).reshape(
                        -1, 1
                    )
                    G_lil[currow : currow + nb_cnst, 1] = np.dot(_F[i], _b[i]).reshape(
                        -1, 1
                    )
                    h[currow : currow + nb_cnst] = _h[i] - np.dot(_F[i], _c[i])
                    currow += nb_cnst

            if _ubound is not None:
                G_lil[currow, 0] = 1
                G_lil[currow + 1, 0] = -1
                h[currow : currow + 2] = [_ubound[i, 1], -_ubound[i, 0]]
                currow += 2

            if _xbound is not None:
                G_lil[currow, 1] = 1
                G_lil[currow + 1, 1] = -1
                h[currow : currow + 2] = [min(ECOS_MAXX, _xbound[i, 1]), -_xbound[i, 0]]
                currow += 2
        ## Fill 4)
        for k in self._conic_idx:
            _a, _b, _c, _P, _, _ = self.params[k]

            # NOTE: Here the following arrangement is used.
            # G_i = [  a_ij        b_ij  ]
            #       [ -P_ij^T[:, :2]     ]
            # h_i = [- c_ij         ]
            #       [- P_ij^T[:, 2] ]
            for j in range(_a.shape[1]):
                G_lil[currow, :] = [_a[i, j], _b[i, j]]
                G_lil[currow + 1 : currow + 4, :] = _P[i, j].T[:, :2]
                h[currow] = -_c[i, j]
                h[currow + 1 : currow + 4] = _P[i, j].T[:, 2]
                currow += 4

        # Fill
        G = scipy.sparse.csc_matrix(G_lil)
        result = ecos.solve(g, G, h, dims, verbose=False)
        accepted_infos = ["Optimal solution found", "Close to optimal solution found"]
        if result["info"]["infostring"] in accepted_infos:
            success = True
        else:
            success = False
            logger.warning(
                "Optimization fails. Result dictionary: \n {:}".format(result)
            )

        ux_opt = np.zeros(2)
        if success:
            ux_opt = result["x"]
        else:
            ux_opt[:] = np.nan
        return ux_opt
