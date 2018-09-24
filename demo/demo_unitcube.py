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
left =  "near(x[0], 0.0) && on_boundary"
right = "near(x[0], 1.0) && on_boundary"

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"), degree=2)
r = Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3, degree=2)

poro.set_solid_boundary_conditions([c, r], [left, right])

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
