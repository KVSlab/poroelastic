__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
sys.path.append("../poroelastic")

import poroelastic as poro
import dolfin as df


comm = df.mpi_comm_world()

nx = 50
mesh = df.UnitSquareMesh(nx, nx)
params = poro.ParamParser("../data/demo_unitcube.cfg")

pprob = poro.PoroelasticProblem(mesh, params)

# Mark boundary subdomians
class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 1.0) and on_boundary

# Sub domain for inflow (right)
class Top(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 1.0) and on_boundary

# Sub domain for outflow (left)
class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 0.0) and on_boundary

boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
right = Right()
top = Top()
bottom = Bottom()
right.mark(boundaries, 1)
top.mark(boundaries, 2)
bottom.mark(boundaries, 3)

# Define Dirichlet boundary conditions
zero = df.Constant(0.0)
dt = params.params["dt"]
squeeze = df.Expression("-1e-2*t", t=0.0, degree=1)

pprob.add_solid_dirichlet_condition(zero, boundaries, 1, n=0)
pprob.add_solid_dirichlet_condition(squeeze, boundaries, 2, n=1, time=True)
# pprob.add_solid_dirichlet_condition(zero, boundaries, 2, n=1)
pprob.add_solid_dirichlet_condition(zero, boundaries, 3, n=1)

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
f1 = df.XDMFFile(comm, '../data/demo_unitsquare/uf.xdmf')
f2 = df.XDMFFile(comm, '../data/demo_unitsquare/mf.xdmf')
f3 = df.XDMFFile(comm, '../data/demo_unitsquare/p.xdmf')
f4 = df.XDMFFile(comm, '../data/demo_unitsquare/du.xdmf')

set_xdmf_parameters(f1)
set_xdmf_parameters(f2)
set_xdmf_parameters(f3)
set_xdmf_parameters(f4)


for Mf, Uf, p, Us, t in pprob.solve():

    dU, L = Us.split()

    poro.write_file(f1, Uf, 'uf', t)
    poro.write_file(f2, Mf, 'mf', t)
    poro.write_file(f3, p, 'p', t)
    poro.write_file(f4, dU, 'du', t)

f1.close()
f2.close()
f3.close()
f4.close()
