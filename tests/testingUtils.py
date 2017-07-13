import numpy as np
from topp.fastTOPP import INFTY, PathConstraint


def canonical_to_TypeI(pc):
    """ Convert a canonical pc to a Type I pc.
    """
    abarnew = pc.a
    bbarnew = pc.b
    cbarnew = pc.c
    Dnew = np.array([np.eye(pc.nm) for i in range(pc.N + 1)])
    lnew = - INFTY * np.ones((pc.N + 1, pc.nm))
    hnew = np.zeros((pc.N + 1, pc.nm))
    return PathConstraint(abar=abarnew, bbar=bbarnew, cbar=cbarnew,
                          D=Dnew, l=lnew, h=hnew, ss=pc.ss)


