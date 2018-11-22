__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
sys.path.append("../poroelastic")

import poroelastic as poro
import dolfin as df


comm = df.mpi_comm_world()

nx = 100
mesh = df.UnitSquareMesh(nx, nx)
params = poro.ParamParser("../data/demo_unitcube.cfg")

pprob = poro.PoroelasticProblem(mesh, params)

# Mark boundary subdomians
bottom = "near(x[1], 0.0) && on_boundary"
top = "near(x[1], 1.0) && on_boundary"

# Define Dirichlet boundary conditions
zero = df.Constant(0.0)
dt = params.params["dt"]
squeeze = df.Expression("-0.01*t", t=0.0, degree=1)

pprob.add_solid_dirichlet_condition(squeeze, top, n=1, time=True)
pprob.add_solid_dirichlet_condition(zero, bottom, n=1)
pprob.add_solid_dirichl

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
f1 = df.XDMFFile(comm, '../data/demo_unitsquare/uf.xdmf')
f2 = df.XDMFFile(comm, '../data/demo_unitsquare/du.xdmf')

set_xdmf_parameters(f1)
set_xdmf_parameters(f2)


for Uf, Us, t in pprob.solve():

    dU, L = Us.split()

    poro.write_file(f1, Uf, 'uf', t)
    poro.write_file(f2, dU, 'du', t)

f1.close()
f2.close()
