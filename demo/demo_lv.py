__author__ = "Alexandra Diem <alexandra@simula.no>"

import poroelastic as poro
import dolfin as df
import matplotlib.pyplot as plt


comm = df.mpi_comm_world()

mf = df.XDMFFile(comm, "./data/lv_mesh/LVH.xdmf")
mesh = df.Mesh()
mf.read(mesh)
mf.close()

params = poro.ParamParser()

markers = {"base": 10, "epicardium": 40, "endocardium": 30}
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bf = df.XDMFFile(comm, "./data/lv_mesh/LVH_mf.xdmf")
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
inflate = df.Constant(1e-3)
vzero = df.Constant((0.0, 0.0, 0.0))
zero = df.Constant(0.0)
dt = params.params["dt"]

pprob = poro.PoroelasticProblem(mesh, params, boundaries=boundaries, markers=markers, territories=territories)
pprob.add_solid_dirichlet_condition(vzero, boundaries, markers["base"])

conditions = {markers["endocardium"]: inflate}
pprob.add_solid_neumann_conditions(conditions)

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
f1 = df.XDMFFile(comm, './data/demo_lv/uf.xdmf')
f2 = df.XDMFFile(comm, './data/demo_lv/mf.xdmf')
f3 = df.XDMFFile(comm, './data/demo_lv/p.xdmf')
f4 = df.XDMFFile(comm, './data/demo_lv/du.xdmf')

set_xdmf_parameters(f1)
set_xdmf_parameters(f2)
set_xdmf_parameters(f3)
set_xdmf_parameters(f4)

def near(a, b):
    tol = 1e-8
    return True if b-tol < a < b+tol else False

dt_save = 0.01
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
