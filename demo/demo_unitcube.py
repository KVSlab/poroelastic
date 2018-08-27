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

bcs = []
poro.set_boundary_conditions(bcs)

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
f1 = XDMFFile(mpi_comm_world(), '../data/demo_unitcube/u1.xdmf')

set_xdmf_parameters(f1)

for U, t in poro.solve():
    dU, L = U.split()

    utils.write_file(f1, dU, 'du', t)

f1.close()
