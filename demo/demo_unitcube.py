__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
sys.path.append("../poroelastic")

import poroelastic

from dolfin import *


comm = mpi_comm_world()

nx = 20
mesh = UnitCubeMesh(nx, nx, nx)
params = ParamParser("demo_parameter.cfg")

poro = PoroelasticProblem(mesh, params)

bcs = []
poro.set_boundary_conditions(bcs)
