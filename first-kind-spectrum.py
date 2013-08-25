import numpy as np
import pyopencl as cl
from pytential.mesh.generation import (  # noqa
        make_curve_mesh, starfish, drop)
import numpy.linalg as la

import logging
logging.basicConfig(level=logging.INFO)

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

target_order = 6
qbx_order = 4
nelements = 20
mode_nr = 4

mesh = make_curve_mesh(starfish,
        np.linspace(0, 1, nelements+1),
        target_order)

from pytential.discretization.qbx import make_upsampling_qbx_discr

discr = make_upsampling_qbx_discr(
        cl_ctx, mesh, target_order, qbx_order)

nodes = discr.nodes().with_queue(queue)

angle = cl.clmath.atan2(nodes[1], nodes[0])

from pytential import bind, sym
representation = sym.S(0, sym.var("sigma"))
op = representation

bc = cl.clmath.cos(mode_nr*angle)

bound_op = bind(discr, op)

from sumpy.tools import build_matrix
mat = build_matrix(bound_op.scipy_op(queue, "sigma"))

w, v = la.eig(mat)

import matplotlib.pyplot as pt
pt.plot(w.real, w.imag, "x")
pt.rc("font", size=20)
pt.grid()
pt.show()
