from Hyperelasticity_Cube import Hyperelastic_Cube


""" Set up for simulation using the poroelastic package. """
import uuid
import sys
from ufl import grad as ufl_grad
import sys
import numpy as np
import dolfin as df
import poroelastic as poro
from poroelastic.material_models import *
import poroelastic.utils as utils



comm = df.mpi_comm_world()
#
#
# Create mesh
#
mesh = df.UnitCubeMesh(16,12,12)
#
#
params = poro.ParamParser()
#
#
data_dir = str(uuid.uuid4())
#
# Add result to parameter dictionary providing section, key, value
#
params.add_data("Simulation", "dir", data_dir)
#
# Print unique simulation ID to screen
#
print("Simulation ID {}".format(data_dir))
#
#
pprob = poro.PoroelasticProblem(mesh, params.p)
#
#
class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 0.0) and on_boundary

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 1.0) and on_boundary

class Top(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 1.0) and on_boundary

class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 0.0) and on_boundary
#
# Initialize mesh function for boundary domains.
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
#
boundaries.set_all(0)
#
# Initialize boundary subdomain instances.
#
left = Left()
right = Right()
top = Top()
bottom = Bottom()
#
# Mark boundary subdomains.
#
left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)
#
# Define Dirichlet boundary (x = 0 or x = 1)
# The Dirichlet boundary values are defined using compiled expressions::
#
zero = df.Constant((0,0,0))
r = df.Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = np.pi/7, degree=2)
#
# Define Dirichlet boundary conditions on boundary subdomains for solid
#
pprob.add_solid_dirichlet_condition(zero, boundaries, 1)
pprob.add_solid_dirichlet_condition(r, boundaries, 2)
#

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

#
# Setting the number of compartments by the integer found for the key 'Parameter' ,
# 'N' in the input file.cfg.
#
N = int(params.p['Parameter']['N'])

f4 = df.XDMFFile(comm, '../data/{}/du.xdmf'.format(data_dir))
#
set_xdmf_parameters(f4)
#
#
# Define new measures associated with exterior boundaries.
dx = df.Measure("dx")
ds = df.Measure("ds")(subdomain_data=boundaries)
#
# Set start variables for the calculations
domain_area = 1.0
#
#
phi = params.p['Parameter']["phi"]
dt = params.p['Parameter']["dt"]
tf = params.p['Parameter']["tf"]
#
u = Hyperelastic_Cube(16,12,12)
for Us, t in pprob.solve():

    dU = Us

    #diff = project(dU-u, dU.function_space())
    poro.write_file(f4, dU, 'du', t)

    domain_area += df.assemble(df.div(dU)*dx)*(1-phi)




f4.close()
#
# error = sum(avg_error)/len(avg_error)
#
#
params.write_config('../data/{}/{}.cfg'.format(data_dir, data_dir))

print("I finished")
#


#u = Hyperelastic_Cube(24,16,16)
error = errornorm(u, dU, 'L2')
print("The error is: {}".format(error))
