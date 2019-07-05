__author__ = "Alexandra Diem, Lisa Pankewitz"
__copyright__ = "Copyright 2019, Alexandra Diem"
__license__ = "BSD-3"
__maintainer__ = "Alexandra Diem"
__email__ = "alexandra@simula.no"

""" This poroelastic demo implements an inflating unit square using the porous
mechanical framework for modelling the interaction between coronary perfusion
and myocardial mechanics based on J Biomech. 2012 Mar 15; 45(5): 850–855.

The demo is implemented in the main python file 'demo_inflate.py' and requires
the package 'poroelastic'. The module 'poroelastic' implements the multicompartment
poroelastic equations. 'poroelastic' requires Python 3.x, and is based on
FEniCS 2017.2.0 (upwards compatibility is suspected, but has not yet been
tested).

It is recommended to setup FEniCS on Docker. A detailed manual for the
installation procedure can be found here https://fenicsproject.org/download/.

In short, to create an image that contains all dependencies of 'poroelastic'
run:

    $docker build --no-cache -t poroelastic:2017.2.0 .

You can then run the docker container using the following command:

    $docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared "poroelastic:2017.2.0"

The tag reflects the FEniCS version used to develop the package.

To view the output Paraview 5.x is required.

"""

# Equation and problem definition - class PoroelasticProblem
# -----------------------------------------------------------
# This demonstration aims to capture the coupling between the fluid flow and the
# mechanical deformation of the myocardial tissue. This is done by a porous
# mechanical framework presented in J Biomech. 2012 Mar 15; 45(5): 850–855.
# The mathematical equations needed for solving the perfusion problem of the left
# ventricle are represented in the 'problem' module.
# To understand the functions and classes behind this demo, we will walk through
# the main functions in the module 'problem'.
#
# To solve the poroelasticity problem described here, the class 'PoroelasticProblem'
# was created. It implements the main functionalities for handling the equations
# of the poroelastic problem.
# To solve the two-way fluid-solid coupling, an iterative scheme is
# performed between the equations of momentum balance of the solid kinematics
# goverened by the constitutive law (I), the compartmental fluid pressure (II)
# and the change fluid mass increase over time (III).
#
# (I) is described in the function 'set_solid_variational_form' of the class
# 'PoroelasticProblem'. It implements the solid variational form of the momentum
# balance equation provided in the paper:
#
#   $ Div (F*S) = 0
#
# where S is the second Piola-Kirchhoff stress tensor subjected to the volume
# constraint given by
#
#   $ S = dPsi/dE + L*J*C^-1
#
# such that F*S is given by
#
#   $ \Psi^{s}_{cons} =  \Psi^{s} + \lambda(J-1- \sum_{i}^{\N}\frac{ m_{i}}{\rho_{f}}$
#
# with L ... Lagrange multiplier enforcing volume constraints
# with m ... sum fluid mass increase over all compartments
# with rho ... fluid density
#
# where 'self.Psi' represents the constitutive law defined by one of the
# material models implemented in the module 'material_models'
#
#    $ self.Psi = self.material.constitutive_law(J=self.J, C=self.C,
#                                                M=m, rho=rho, phi=phi0)
#
# with J ... determinant of the deformation gradient tensor 'F'
# with C ... right Cauchy-Green deformation tensor
# with M ... total fluid mass increase
# with rho ... fluid density
# with phi ... porosity of the fluid phase
#
# The variational form of the solid momentum balance equation is implemented
# in 'set_solid_variational_form' as
#
#   $ Form = derivative(Psic, U, V)
#
# for 'Function' 'U' and 'TestFunction' 'V' on 'FunctionSpace' 'self.FS_S',
# with Jacobian
#
#   $ dF = derivative(Form, U, TrialFunction(self.FS_S))
#
# (II) The 'fluid-solid-coupling' governing the compartmental fluid pressure
# is repesented by the following equation
#
#   $ p_{i} = \frac{\partial \Psi^{s}}{\partial (J \phi_{f,i})} - \lambda $
#
# which is implemented as a variational form by solving
#
#   $ p*q*dx = (tr(diff(self.Psi, self.F) * self.F.T))/self.phif[i]*q*dx - L*q*dx
#
# with q ... 'TestFunction' on 'FunctionSpace' 'self.FS_F'
# with p ... 'TrialFunction' on 'FunctionSpace' 'self.FS_F'
#
# The function spaces are defined in the function 'create_function_spaces'
# as 'MixedElement' space if there is more than one compartment, i.e N>1.
#
# (III) The fluid variational form for multiple compartments (N>1) is
# implemented in the function 'set_fluid_variational_form' as
#
#     Form = sum([k*(m[i] - m_n[i])*vm[i]*dx for i in range(self.N)])\
#         + sum([dot(grad(M[i]), k*(dU-dU_n))*vm[i]*dx
#                                             for i in range(self.N)])\
#         + sum([inner(-((rho * self.J * inv(self.F) * self.K() * inv(self.F.T))*grad(self.p[i]), grad(vm[i]))*dx
#                                             for i in range(self.N)])
#
# with m ... (mixed) 'TrialFunction' on 'FunctionSpace' 'self.FS_M'
# with vm ... (mixed) 'TestFunction' on 'FunctionSpace' 'self.FS_M'
#
# Time is discretised according to the theta-rule (Crank-Nicolson) by defining
#
#   $M = th*m + th_*m_n
#
# with m_n ... solution for m from the previous time step
#
# where the inflow and outflow terms are represented by:
#
#     Form += -rho*self.qi*vm*dx
#
#     Form += rho*q_out*vm*dx
#
# and the compartment exchange is included by:
#
#     for i in range(len(beta)):
#         Form += -self.J*beta[i]*((self.p[i] - self.p[i+1])*vm[i] +\
#                                 (self.p[i+1] - self.p[i])*vm[i+1])*dx
#
#
# Implementation
# --------------
#
# This description goes through the implementation of 'demo_inflate.py', which
# implements the inflation of a unit square using the fluid source term q
# step-by-step.
#
# First the required modules are imported.
#
import sys
import uuid
import poroelastic as poro
import dolfin as df
import numpy as np
#
# Access MPI communicator using dolfin's provided MPI interface to enable
# parallel computing.
#
comm = df.mpi_comm_world()
#
# We need to create a mesh based on the dolfin class dolfin.cpp.Mesh. Here the
# mesh consists of a unit square discretised into 2 x nx x nx triangles.
#
# Create mesh
nx = 10
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
# Next, we would like to create a meshfunction allowing for the storage
# and numbering of subdomains using the 'MeshFunction' from the dolfin package.
# When creating a MeshFunction an argument defining the type of MeshFunction
# is required. This is represented by the first argument which in our example
# is defined by 'size_t'. 'size_t' defines that an integer is taken as argument
# and all facets will as consequence be given this index.
# The second argument which is optional, defines the mesh.
# The third argument provides the topological dimension of the mesh which in
# our case is '-1'. This argument is optional, but important to be defined in
# respect to the boundary conditions, which need to be a dimension lower than
# the space we are working in.
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
# We set markers allowing for tracking of changes in mesh before the mesh is deformed.
# Often, you will find that these markers are saved into a dictionary.
#
left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)
#
# Next, we need to add the dirichlet boundary conditions for the solid.
# For that we will use the function 'add_solid_dirichlet_condition' from the
# class 'PoroelasticProblem' from the 'Poroelastic' package.
# This function allows for setting different dirichlet boundary conditions
# depending on the subspace.
#
# The boundaries are set with the first argument in correspondence to the subdomains
# marked by 1 (Left) and 4 (Bottom) defined by the boundaries function.
# The third argument represents the sub domain instance the condition is
# defined for.
# In our example the fourth argument functions as a keyword argument setting
# the value for n in the vectorspace. If intending to set a boundary for a complete vectorspace,
# one would pass the argument defining the boundary as vector and not require
# to define the single element in the vectorspace by 'n'. 'n' would not be passed
# as argument.
# The boundary conditions are stored in a dictionary.
# Optionally, a string specifying a DirichletBC method can be passed as an argument.
# This allows for the usage of DirichletBC function defined methods provided
# by the dolfin package.
# A timestep condition can be enforced by adding 'time' which functions as a boolean
# as an argument. For usage 'tcond' can be defined as an additional variable referring to the
# to an expression defining the condition of the parameter 't'. The second argument
# in tcond would define the current time setting and the third could refer to the
# degree of the expression.
#
# example:
#   tcond = df.Expression('t', t=0.0, degree=1)
#   pprob.add_solid_dirichlet_condition(tcond, boundaries, 1, time=True)
#
# Time conditions will then be saved in an additiona new list.
#
# Define Dirichlet boundary conditions
zero = df.Constant(0.0)
pprob.add_solid_dirichlet_condition(zero, boundaries, 1, n=0)
pprob.add_solid_dirichlet_condition(zero, boundaries, 4, n=1)
#
# Eventually, we will have to store the data produced in files.
# For that to work we will use a dolfin class supporting the output of meshes
# and functions in XDMF format. This will allow us to create an XML file describing
# the data produced and pointing to a so-called HDF5 file that will store the actual
# data.
# In order to allow output of data in parallel, 'comm ' will be used as an argument.
# The second argument passed represents the location the file will be stored in.
#
# Before storing the data, we need to define major parameters of the xdmf files.
# The function 'set_xdmf_parameters' sets functionalities for processing and opening
# of the XDMFFiles.

#1) ' flush_output'
# Enables the functionality to preview the XDMFFile produced, which comes in handy
# when you have an iterative process taking a long time to run and you want to check
# whether the results provided meet your expectations. If this is not set to true,
# you cannot read the file produced (e.g. with Paraview) until the program terminates.

#2) 'functions_share_mesh'
# When enabled it makes all functions share the same mesh and time series.

#3) 'rewrite_function_mesh'
#  If False, this parameter limits each function to one mesh for the complete time
# series. The Mesh will not be rewritten every time-step.

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
f2 = [df.XDMFFile(comm, '../data/{}/mf.xdmf'.format(data_dir,i)) for i in range(N)]
f3 = [df.XDMFFile(comm, '../data/{}/p{}.xdmf'.format(data_dir, i)) for i in range(N)]
f4 = df.XDMFFile(comm, '../data/{}/du.xdmf'.format(data_dir))
#
# Initialize 'set_xdmf_parameters' for XDMFFiles to be created
#
[set_xdmf_parameters(f1[i]) for i in range(N)]
[set_xdmf_parameters(f2[i]) for i in range(N)]
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
[f2[i].close() for i in range(N)]
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
