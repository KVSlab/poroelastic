__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
sys.path.append("../poroelastic")

import poroelastic as poro
import dolfin as df


comm = df.mpi_comm_world()

nx = 100
mesh = df.UnitSquareMesh(nx, nx)
params = poro.ParamParser()

pprob = poro.PoroelasticProblem(mesh, params)

# Mark boundary subdomians
class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 0.0) and on_boundary

class Top(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 1.0) and on_boundary

class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 0.0) and on_boundary

boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left = Left()
top = Top()
bottom = Bottom()
left.mark(boundaries, 1)
top.mark(boundaries, 2)
bottom.mark(boundaries, 3)

# Define Dirichlet boundary conditions
zero = df.Constant(0.0)
vzero = df.Constant((0.0, 0.0))
squeeze = df.Constant(-1e-2)
dt = params.params["dt"]

pprob.add_solid_dirichlet_condition(zero, boundaries, 1, n=0)
pprob.add_solid_dirichlet_condition(zero, boundaries, 3, n=1)
pprob.add_solid_dirichlet_condition(squeeze, boundaries, 2, n=1)

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

    dU, L = Us.split(True)

    poro.write_file(f1, Uf, 'uf', t)
    poro.write_file(f2, Mf, 'mf', t)
    poro.write_file(f3, p, 'p', t)
    poro.write_file(f4, dU, 'du', t)

f1.close()
f2.close()
f3.close()
f4.close()
