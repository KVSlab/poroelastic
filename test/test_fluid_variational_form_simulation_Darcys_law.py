"""
    Simulation testing the function set_fluid_variational_form of the poroelastic package. Simulation
    with fixed dirichlet solid boundary conditions allows comparing output with Darcy's law."""


""" Set up for simulation using the poroelastic package. """
import sys
import uuid
import poroelastic as poro
import dolfin as df
import numpy as np
#import matplotlib.pyplot as plt
#
#
comm = df.mpi_comm_world()
#
# We need to create a mesh based on the dolfin class dolfin.cpp.Mesh. Here the
# mesh consists of a unit square discretised into 2 x nx x nx triangles.
#
# Create mesh
nx = 20
mesh = df.UnitSquareMesh(nx, nx)
#
# Parameters are loaded using the class 'ParamParser' that reads and processes
# the provided cfg file.
#
params = poro.ParamParser()
#
# Next, we generate a unique directory for data storage.
#
# Using the uuid4() function for generating a random UUID (Universal Unique
# Identifier, 128 bit number) a random identifictaion number is created and
# with the string method converted to a string of hex digits.
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
# Next, we initialise the 'PoroelasticProblem' with the Mesh object and ParamParser
# dictionary.
#
pprob = poro.PoroelasticProblem(mesh, params.p)
#
# Next we divide our left ventricle into 4 main subdomains, having their
# individually set boundary conditions.
# For that to work, we create classes for defining parts of the boundaries and
# the interior of the domains.
# We consider Dirichlet boundary conditions. These can be implied by creating a
# simple function returning a boolean. The idea is to return 'True' for points
# inside the subdomain and 'False' for those oustide.
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
# To ensure all boundaries are set to 0 allowing for knowing and tracking
# th exact coordinates, the boundary values need to be set to 0.
#
# Hint: when ommiting this step in older dolfin versions you might actually end
# up with randomly set boundary values.
#
boundaries.set_all(0)
#
# Initialize sub-domain instances.
#
left = Left()
right = Right()
top = Top()
bottom = Bottom()
#
# Next, we initialize the mesh function for boundary domains in sub-domains.
#
left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)
#
# Define Dirichlet boundary conditions for solid, allows for focusing on fluid problem comparable to darcy flow
zero = df.Constant((0,0))
pprob.add_solid_dirichlet_condition(zero, boundaries, 1)
pprob.add_solid_dirichlet_condition(zero, boundaries, 2)
pprob.add_solid_dirichlet_condition(zero, boundaries, 3)
pprob.add_solid_dirichlet_condition(zero, boundaries, 4)
#
# Dirichlet BC left fluid mass 1e-3, set kwargs time=False, no Source needed since one compartment (N=1)
# without setting condition , already 0 Neumann conditions
# allows for pressure gradient
half = df.Constant(1e-3)
zero_scalar = df.Constant(0.88)
pprob.add_fluid_dirichlet_condition(half, boundaries, 1, time=False)
#need to define right dirichlet boundary condition as string in .cfg file
# currently set to :  "1e-6*(1-x[0])"
#pprob.add_fluid_dirichlet_condition(zero_scalar, boundaries, 2, time=False)
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
#
f1 = [df.XDMFFile(comm, '../data/{}/uf{}.xdmf'.format(data_dir, i)) for i in range(N)]
f2 = df.XDMFFile(comm, '../data/{}/mf.xdmf'.format(data_dir))
f3 = [df.XDMFFile(comm, '../data/{}/p{}.xdmf'.format(data_dir, i)) for i in range(N)]
f4 = df.XDMFFile(comm, '../data/{}/du.xdmf'.format(data_dir))
#
# Initialize 'set_xdmf_parameters' for XDMFFiles to be created
#
[set_xdmf_parameters(f1[i]) for i in range(N)]
set_xdmf_parameters(f2)
[set_xdmf_parameters(f3[i]) for i in range(N)]
set_xdmf_parameters(f4)
#
#
# Define new measures associated with exterior boundaries.
dx = df.Measure("dx")
ds = df.Measure("ds")(subdomain_data=boundaries)
#
# Set start variables for the calculations
sum_fluid_mass = 0
theor_fluid_mass = 0
sum_disp = 0
domain_area = 1.0
#
#
phi = params.p['Parameter']["phi"]
rho = params.p['Parameter']["rho"]
qi = params.p['Parameter']["qi"]
dt = params.p['Parameter']["dt"]
tf = params.p['Parameter']["tf"]
#
# The average error which will be calculated is stored in the list 'avg_error'.
avg_error = []
#
#
for Mf, Uf, p, Us, t in pprob.solve():

    dU, L = Us.split(True)

    [poro.write_file(f1[i], Uf[i], 'uf{}'.format(i), t) for i in range(N)]
    poro.write_file(f2, Mf, 'mf', t)
    [poro.write_file(f3[i], p[i], 'p{}'.format(i), t) for i in range(N)]
    poro.write_file(f4, dU, 'du', t)

    domain_area += df.assemble(df.div(dU)*dx)*(1-phi)
    sum_fluid_mass += df.assemble(Mf*dx)
    # No calculation of theor fluid with qi since qi=0 since only using source term
    #theor_fluid_mass += qi*rho*dt
    #theor_sol = theor_fluid_mass*domain_area
    sum_disp += df.assemble(dU[0]*ds(4))
    #avg_error.append(np.sqrt(((df.assemble(Mf*dx)-theor_sol)/theor_sol)**2))

    #print(theor_sol, df.assemble(Mf*dx))

[f1[i].close() for i in range(N)]
f2.close()
[f3[i].close() for i in range(N)]
f4.close()
#
# error = sum(avg_error)/len(avg_error)
#
#
params.write_config('../data/{}/{}.cfg'.format(data_dir, data_dir))
#
# Finally, the result for the expected sum fluid mass, the calculated sum of the
# fluid mass and the average error over all time steps are ptinted to the screen.
#
#print("Expected sum fluid mass: {}".format(theor_fluid_mass))
print("Sum fluid mass: {}".format(sum_fluid_mass))

#print("Average error over all time steps: {}".format(error))
#print("Pressure: {}".format(p[i] for i in range(N)))
