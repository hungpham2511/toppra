%load_ext autoreload
%autoreload 2

import toppra
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

p = toppra.SimplePath(np.r_[0, 1, 2], np.r_[0, 1, 0])
xs = np.linspace(*p.path_interval, 200)
plt.plot(xs, p(xs))
plt.show()


f = si.BPoly.from_derivatives([0,0.75,2],[[0, 0], [1, 0], [0, 0]], orders=2)
# f = si.CubicHermiteSpline([0,1,2],[0, 1, 0], [0, 0, 0])
x = np.linspace(0, 2, 200)
plt.plot(x, f(x, 1))
plt.show()
