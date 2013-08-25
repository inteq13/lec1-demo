import numpy as np
import pyopencl as cl
from pytential.mesh.generation import (  # noqa
        make_curve_mesh, starfish, drop)
from sumpy.visualization import FieldPlotter

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
representation = sym.D(0, sym.var("sigma"))
op = representation - 0.5*sym.var("sigma")

bc = cl.clmath.cos(mode_nr*angle)

bound_op = bind(discr, op)
from pytential.gmres import gmres
gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma"),
        bc, tol=1e-14, progress=True,
        hard_failure=True)

import sys
sys.exit()

sigma = gmres_result.solution

fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)
from pytential.discretization.target import PointsTarget
fld_in_vol = bind(
        (discr, PointsTarget(fplot.points)),
        representation)(queue, sigma=sigma).get()

fld_on_bdry = bound_op(queue, sigma=sigma).get()

from mayavi import mlab
fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)

mlab.colorbar()
mlab.show()
