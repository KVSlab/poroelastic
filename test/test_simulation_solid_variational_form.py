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
#
# f1 - list divergence stress
# f2 - mass fluid
# f3 - pressure
 # f4 - list scalar of divergence change of deformation solid
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
rho = params.p['Parameter']["rho"]
qi = params.p['Parameter']["qi"]
dt = params.p['Parameter']["dt"]
tf = params.p['Parameter']["tf"]
#
# The average error which will be calculated is stored in the list 'avg_error'.
avg_error = []
#
# To solve the variational problem the class 'PoroelasticProblem' defined in the
# the module 'problem' in the package 'Poroelastic' provides the function 'solve()'.
# The function allows for parallel computing and will return process rank of the
# computed processes for the communicator or the local machine.
# Furthermore, the function initializes the tolerance function TOL() from the
# class 'PoroelasticProblem' and sets it to the value found for the keys 'TOL'
# in the dictionary created when reading in the configuration file.
# The maximum number of iterations is set to 100.
# The current time is set to t = 0.0.
# dt is initialized as a constant of the function 'set_fluid_variational_form'
# of the class 'Poroelastic'.
# The solve() function initiates solving of the fluid mass (Mf), fluid divergence (Uf),
# the pressure (p), the solid divergence (Us) and the time (t).
# In the solve() function the NonlinearVariationalProblem for the solid as well as
# the fluid phase will be executed.  For that, as described above, the
# variational problems are expressed in there variational form stored in
# the 'set_fluid_variational_form' and the 'set_solid_variational_form' function
# provided by the module 'problem' in the 'Poroelastic' package.
# The variational forms are in the 'solve' functions passed as parameter of the
# dolfin class 'NonlinearVariationalProblem', a class representing a nonlinear
# variational problem.
# Besides the variational form, the unknown function has to be passed as parameter,
# which in our example is represented by mf (fluid mass) or Us (extended constitutive law).
# The other parameters passed to specify boundary conditions and the Jacobiany
# and the Jacobian are optional.
# After setting the parameters of the 'NonlinearVariationalProblem' the 'choose_solver'
# function is initated out of the 'solve' function.
#
# The 'choose_solver' function of the 'problem' module allows the user to choose
# between a direct solver and an iterative one. For that the 'choose_solver' functions
# goes through the params dictionary created when reading in the configuration file,
# and if there is a key defined for ' Simulation and solver' defining the value as
# directed the method of solving the NonlinearVariationalProblem will be chosen to
# be done directly by executing the function 'direct_solver'.
# if not running a Simulation, the value and method ' direct' can be ommitted
# and the variational problem can be solved in iterative manner by executing the
# function 'iterative_solver'.
#
# A main difference between the two approaches is the computational expense. while
# we will use in this demo the iterative approach, which is less computationally
# expensive, one could also define in the configuration file the use of the direct
# approach. While the the direct approach allows solving the problem in one major
# computational step, requiring a lot of RAM, the indirect method approaches the
# solution gradually by in smaller steps which require less RAM but create the
# need for iteration over solved steps.
# The iterative approach has in consequence the advantage of being faster.
# Nevertheless, the tolerance estimate wil be defined by the  solution from the
# direct method of the well-defined or well-conditioned problem.
#
# A while-loop in the 'solve' function enables ffor running the Simulation
# for as long as defined in the 'Params' dictionary created when reading the
# configuration file.
# The MPI-worker number 1 (rank 0) is then defined to be in charge of printing
# the time by initiating the function 'print_time'.
# 'print_time' is a function defined in the 'utils' module of the 'Poroelastic'
# package.
# The variables 'iter' (iteration) and 'eps' (error-per-step) are initialized.
#
# A second while-loop in the 'solve' function limits the number of maximum_iterations
# to eps > tol and iter < maxiter. That means, that the iteration over the equations
# is done as long as the error per step (eps) is above the threshold for the tolerance
# (tol) and the maximum number of iterations set has not been exceeded.
#
# During each iteration the latest calculated pressure function 'p' is assigned to
# the function 'mf_'. For that, the dolfin function .assign() is used, allowing
# to assign one function to another.
# The variational problem for the solid, for the fluid and for the fluid_solid_coupling
# (defining 'p') are initiated respectively.
# The error variable 'e' is then calculated by first substratingthe pressure variable
# 'mf' assigned from the iteration step before from the current 'p[0]' .
# The error 'eps' is in the next step evaluated by the square of the error "e"
# following the L2 -norm.
# If the condition eps > tol and iter < maxiter still apply, the while loop of the
# 'solve' function is not broken.
# After breaking out of the itaeration determining while - loop, the current Solution
# for the functions 'mf' and 'Us' are stored as the previous solutions
# using the dolfin.assign() function.
#
# In the next step, the Lagrangian Darcy flow vector, in short referred to as fluid vector,
# is calculated by initiating the function 'calculate_flow_vector' of the
# class 'PoroelasticProblem'. For solving this variational problem, the 'solver_parameters'
# are set to the iterative solver 'minres' with the precondiitoner 'hypre_amg'
# the same as set as default in the 'iterative_solver' function.
# The 'solve' function, or in this case even defined as a python generator by using the
# 'yield' statement instead of the 'return' statement, yields the objects
# 'self.mf', 'self.Uf', 'self.p', 'self.Us', 't'.
# 'yield' allows for the local variables being created to be kept and prevents
# the function from exiting. This way for the simulation, yield allows us to
# produce a sequence of values. This facilitates iterating over the sequence of solutions created,
# but does not require us to store the entire sequence in memory as the 'return'
# statement would.
# In the next step the mesh is moved by initiating the 'move_mesh' function.
# The 'move_mesh' function is defined in the 'Problem' module. It takes advantage
# of the 'ALE.move' class returning the projection of the components of the function
# 'dU' onto the VectorFunctionSpace.
# In the last step the the time print statement is updated by adding dt to the current
# time.
# An additional print statement after exiting the while-loop is added to avoid
# overwriting the time print statements when the next output is printed.
#
# Moving back to the 'demo_inflate' , we are using the dolfin provided 'split()'
# function to extract the subfunctions of 'Us' extracting sub functions.
#
# The results will be written to the XDMFFiles created earlier, using the 'write_file'
# function of the 'utils' module. This function itself will initiate the DOLFIN
# provided 'set_log_level' function, deciding which messages routed through the
# logging system will be printed to the console. Calling the function 'set_log_level',
# we can specify the log level of the messages printed by setting the value for the
# optional integer argument.
# In our example it is set to 40, meaning with the default level being 20, only
# messages higher than or equal to the set log level will be printed.
# next, the 'write_checkpoint' allows for saving a function to an XDMFFile for
# checkpointing, taking in the parameters of the function to save, the name (label)
# of the function used, and the time step.
# Last, the log level is increased by setting the integer to 30, allowing for
# for messages to be printed.
#
# Next in the for loop, the solutions for the domain_area, the sum of the fluid mass
# (sum_fluid_mass), the theoretical fluid mass (theor_fluid_mass), the theoretical
# solution for the fluid mass in the domain (theor_sol), the sum of the dispersion
# (sum_disp)  and the average error (avg_error) are computed.
# The use of the dolfin function 'assemble' returns depending on the input,
# a scalar value, a vector, a matrix or a higher rank tensor (in our case a scalar
# or a matrix).
#
# The 'avg_error' saves the error according errornorm L2 and normalized by the
# theoretical solution 'theor_sol'. The 'avg_error' values are appended to a list.
#
# As long as the for loop continues, the theoretical solution and the currently
# approximated solution of the sum of the fluid mass are printed to the screen.
#
# Upon exiting the for loop, the XDMFFiles created are closed by calling the
# 'close()' function.

for Mf, Uf, p, Us, t in pprob.solve():

    dU, L = Us.split(True)

    [poro.write_file(f1[i], Uf[i], 'uf{}'.format(i), t) for i in range(N)]
    poro.write_file(f2, Mf, 'mf', t)
    [poro.write_file(f3[i], p[i], 'p{}'.format(i), t) for i in range(N)]
    poro.write_file(f4, dU, 'du', t)

    domain_area += df.assemble(df.div(dU)*dx)*(1-phi)
    sum_fluid_mass += df.assemble(Mf*dx)
    theor_fluid_mass += qi*rho*dt
    theor_sol = theor_fluid_mass*domain_area
    sum_disp += df.assemble(dU[0]*ds(4))
    avg_error.append(np.sqrt(((df.assemble(Mf*dx)-theor_sol)/theor_sol)**2))
    print(theor_sol, df.assemble(Mf*dx))

[f1[i].close() for i in range(N)]
f2.close()
[f3[i].close() for i in range(N)]
f4.close()
#
# The final error is calculated by normalizing the avg_error by the number of elements
# in the list of errors.
#
error = sum(avg_error)/len(avg_error)
#
# The function 'write_config' inherited by the class 'ParamParser' of the module
# param_parser is executed on the configuration files to be created.
#
params.write_config('../data/{}/{}.cfg'.format(data_dir, data_dir))
#
# Finally, the result for the expected sum fluid mass, the calculated sum of the
# fluid mass and the average error over all time steps are ptinted to the screen.
#
print("Expected sum fluid mass: {}".format(theor_fluid_mass))
print("Sum fluid mass: {}".format(sum_fluid_mass))
print("Average error over all time steps: {}".format(error))

print("I finished")
#


u = Hyperelastic_Cube(16,12,12)
error = errornorm(u, dU, 'L2')
print("The error is: {}".format(error))
