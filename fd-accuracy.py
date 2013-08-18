from __future__ import division
import numpy as np
import numpy.linalg as la


c = 20*2*np.pi


def f(x):
    return np.sin(c*x)


def df(x):
    return c*np.cos(c*x)


h_values = []
err_values = []

for n_exp in xrange(5, 24):
    n = 2**n_exp
    h = (1/n)

    x = np.linspace(0, 1, n, endpoint=False).astype(np.float32)

    fx = f(x)
    dfx = df(x)

    dfx_num = (np.roll(fx, -1) - np.roll(fx, 1)) / (2*h)

    err = la.norm(dfx - dfx_num) / la.norm(fx)

    print h, err

    h_values.append(h)
    err_values.append(err)

import matplotlib.pyplot as pt
pt.rc("font", size=16)
pt.title(r"Single precision FD error on $\sin(20\cdot 2\pi)$")
pt.xlabel(r"$h$")
pt.ylabel(r"Rel. Error")
pt.loglog(h_values, err_values)
pt.savefig("fd-error.pdf")
