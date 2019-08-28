__author__ = "Alexandra Diem, Lisa Pankewitz"
__copyright__ = "Copyright 2019, Alexandra Diem"
__license__ = "BSD-3"
__maintainer__ = "Alexandra Diem"
__email__ = "alexandra@simula.no"


# import function based on the demo_hyperelasticity from fenics to evaluate the
# deformation of the solid
from Hyperelasticity_Cube import Hyperelastic_Cube
# import packages needed
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
# Initiate the ParamParser function to read in the parameters given.
params = poro.ParamParser(cfgfile='test_simulation_solid_variational_form.cfg')
#
# Creating a unique name identifying the directiory created, which stores the data.
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
# In our case this means for 'Left' we set the boundaries to x=0.0 .
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
# The Dirichlet boundary values are defined using compiled expressions::
# The left boundary acts as a clamp on the cubs, whereas the right boundary condition
# allows to twist the cube.
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
#
 # --------------------------------------------------------------------------------------
 # Important note: To actually find the output data in the 'data directory' you need
 # to run the script in the directory a hierarchy up the data directory, so that the data
 # directory can actually be found.
 # --------------------------------------------------------------------------------------
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
# dx and ds are predefined measures in dolfin referring to the integration over
# cells and exterior facets (facets on the boundary), respectively.
# Since dx and ds can take additional integer arguments, integration over subdomains
# can be defined by using different variables or integer labels as arguments.
# In order to map the geometry information stored in the mesh functions to the
# measures, we will define new measures and for ds we will use the boundary defining
# mesh function for the subdomains as input.
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
# Using the get_params function provided by the ParamParser class, the configuration
# file provided in self.prams is read and the parameters are stored as dictionary
# params['name'] = value .
# In this instance the key 'parameter' is looked up.
#
phi = params.p['Parameter']["phi"]
qi = params.p['Parameter']["qi"]
dt = params.p['Parameter']["dt"]
tf = params.p['Parameter']["tf"]
#
# Calculating the solution for the Hyperelastic_Cube using the exact same mesh(object)
# as used in the simulation.
u = Hyperelastic_Cube(mesh)
for Mf, Uf, p, Us, t in pprob.solve():

    dU, L = Us.split(True)

    [poro.write_file(f1[i], Uf[i], 'uf{}'.format(i), t) for i in range(N)]
    poro.write_file(f2, Mf, 'mf', t)
    [poro.write_file(f3[i], p[i], 'p{}'.format(i), t) for i in range(N)]
    poro.write_file(f4, dU, 'du', t)

# Commented out below  is the option for creating a ParaView readable file to visualize the difference
# between the solution of the two approaches over the surface of the cube.
# The error values over the elememts of the mesh is expected to vary and cause error in the magnitude
# of up to 3e-3. This observation is due to the difference in the calculation of the solution for the solid.
# In addition to the constitutive law, the PoroelasticProblem applies the Lagrange multiplier
# subjecting the constitutive law to the constraint evoked by the influence of the fluid mass divergence,
# density of the fluid and in sum the difference of the fluid influence in each compartment
# changing the determinante J.
# This difference in calculation leads to the error observed.


    #diff = project(dU-u, dU.function_space())
    #poro.write_file(f4, diff, 'du', t)

    domain_area += df.assemble(df.div(dU)*dx)*(1-phi)
    sum_fluid_mass += df.assemble(Mf*dx)
    sum_disp += df.assemble(dU[0]*ds(4))


[f1[i].close() for i in range(N)]
f2.close()
[f3[i].close() for i in range(N)]
f4.close()
#
# The function 'write_config' inherited by the class 'ParamParser' of the module
# param_parser is executed on the configuration files to be created.
#
params.write_config('../data/{}/{}.cfg'.format(data_dir, data_dir))
#
# Finally, the result for the expected sum fluid mass, the error between the
# Hyperelastic_Cube solution and the PoroelasticProblem solution according to
# the L2 norm are calculated and printed to the screen.

print("Sum fluid mass: {}".format(sum_fluid_mass))

error = errornorm(u, dU, 'L2')
print("The error is: {}".format(error))
