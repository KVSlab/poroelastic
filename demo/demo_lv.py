__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
sys.path.append("../poroelastic")

import poroelastic as poro
import dolfin as df
import matplotlib.pyplot as plt


comm = df.mpi_comm_world()

mf = df.XDMFFile(comm, "../data/mesh_highres/LV.xdmf")
mesh = df.Mesh()
mf.read(mesh)
mf.close()

params = poro.ParamParser("../data/demo_lv.cfg")

markers = {"base": 1, "epicardium": 2, "endocardium": 3}
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bf = df.XDMFFile(comm, "../data/mesh_highres/LV_mf.xdmf")
bf.read(boundaries)
bf.close()

territories = df.MeshFunction("size_t", mesh, mesh.topology().dim())
territories.set_all(0)

class T1(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[2] < 0.0 + df.DOLFIN_EPS

t1 = T1()
t1.mark(territories, 1)

# Define Dirichlet boundary conditions
inflate = df.Constant(1e-2)
vzero = df.Constant((0.0, 0.0, 0.0))
zero = df.Constant(0.0)
dt = params.params["dt"]

pprob = poro.PoroelasticProblem(mesh, params, boundaries=boundaries, markers=markers, territories=territories)
pprob.add_solid_dirichlet_condition(vzero, boundaries, markers["base"])
pprob.add_solid_dirichlet_condition(vzero, "on_boundary and x[0] > 4.66")
# pprob.add_solid_dirichlet_condition(vzero, "sqrt(x[1]*x[1] + x[2]*x[2]) > 2.5 and x[0] < 0.0", method="pointwise")

conditions = {markers["endocardium"]: inflate}
pprob.add_solid_neumann_conditions(conditions)

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
f1 = df.XDMFFile(comm, '../data/demo_lv/uf.xdmf')
f2 = df.XDMFFile(comm, '../data/demo_lv/mf.xdmf')
f3 = df.XDMFFile(comm, '../data/demo_lv/p.xdmf')
f4 = df.XDMFFile(comm, '../data/demo_lv/du.xdmf')

set_xdmf_parameters(f1)
set_xdmf_parameters(f2)
set_xdmf_parameters(f3)
set_xdmf_parameters(f4)

def near(a, b):
    tol = 1e-10
    return True if b-tol < a < b+tol else False

dt_save = 0.05
t_save = 0.0
for Mf, Uf, p, Us, t in pprob.solve():

    dU, L = Us.split(True)

    if near(t, t_save):
        t_save += dt_save
        poro.write_file(f1, Uf, 'uf', t)
        poro.write_file(f2, Mf, 'mf', t)
        poro.write_file(f3, p, 'p', t)
        poro.write_file(f4, dU, 'du', t)


f1.close()
f2.close()
f3.close()
f4.close()
