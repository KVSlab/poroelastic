__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
sys.path.append("../poroelastic")

from problem import PoroelasticProblem
from param_parser import *
import utils

from dolfin import *


comm = mpi_comm_world()

nx = 20
mesh = UnitCubeMesh(nx, nx, nx)
params = ParamParser("../data/demo_unitcube.cfg")

poro = PoroelasticProblem(mesh, params)

# Mark boundary subdomians
left = "near(x[0], 0.0) && on_boundary"
front = "near(x[1], 0.0) && on_boundary"
bottom = "near(x[2], 0.0) && on_boundary"
top = "near(x[2], 1.0) && on_boundary"

# Define Dirichlet boundary conditions
zero = Constant(0.0)
dt = params.params["dt"]
squeeze = Expression("-0.01*t*x[0]", t=0.0, degree=2)

poro.add_solid_dirichlet_condition(zero, left, n=0)
poro.add_solid_dirichlet_condition(zero, front, n=1)
poro.add_solid_dirichlet_condition(zero, bottom, n=2)
poro.add_solid_t_dirichlet_condition(squeeze, top, n=2)

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
f1 = XDMFFile(mpi_comm_world(), '../data/demo_unitcube/uf.xdmf')
f2 = XDMFFile(mpi_comm_world(), '../data/demo_unitcube/du.xdmf')

set_xdmf_parameters(f1)
set_xdmf_parameters(f2)


for Uf, Us, t in poro.solve():

    dU, L = Us.split()

    utils.write_file(f1, Uf, 'uf', t)
    utils.write_file(f2, dU, 'du', t)

f1.close()
f2.close()
