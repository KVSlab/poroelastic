__author__ = "Alexandra Diem, Lisa Pankewitz"
__copyright__ = "Copyright 2019, Alexandra Diem"
__license__ = "BSD-3"
__maintainer__ = "Alexandra Diem"
__email__ = "alexandra@simula.no"

""" This poroelastic simulation implements a minimal version of the 'problem'
module of the package 'poroelastic'. Using a minimal version ' FluidelasticProblem'
only taking into account the fluid eauations as well as the solid fluid coupling,
it demonstrates in an isolated manner the ability of the
package to illustrate purely fluid related issues and compares this functionality
to the darcy flow in a unit square. The Darcy flow is implemented in the
function 'Darcy()' in the module 'Darcy_Function'.
This simulation functions solely as a test validaing the equations of the package
poroelastic describing the fluid.

The test simulation is implemented in the main python file
'Minimal_fluid_variational_form.py' importing the module 'Darcy_Function'
and requires the package 'poroelastic'. The module 'poroelastic' implements the
multicompartment poroelastic equations. 'poroelastic' requires Python 3.x, and is based on
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


import dolfin as df
import uuid
from ufl import grad as ufl_grad
import sys
import numpy as np
import poroelastic as poro
from poroelastic.material_models import *
import poroelastic.utils as utils
from Darcy_Function import Darcy
import matplotlib.pyplot as plt

# Compiler parameters
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["allow_extrapolation"] = True

set_log_level(30)


class FluidelasticProblem(object):
    """
    Boundary marker labels:
    - inflow (Neumann BC in fluid mass increase)
    - outflow (Neumann BC in fluid mass increase)
    """

    def __init__(self, mesh, params, boundaries=None, markers={}):
        self.mesh = mesh
        self.params = params
        self.markers = markers
        self.N = int(self.params['Parameter']['N'])

        if boundaries != None:
            self.ds = ds(subdomain_data=boundaries)
        else:
            self.ds = ds()

        # Create function spaces
        self.FS_M, self.FS_F, self.FS_V = self.create_function_spaces()

        # Create solution functions
        self.mf = Function(self.FS_M)
        self.mf_n = Function(self.FS_M)
        self.Uf = [Function(self.FS_V) for i in range(self.N)]
        if self.N == 1:
            self.p = [Function(self.FS_M)]

        rho = self.rho()

        phi0 = self.phi()
        if self.N == 1:
            self.phif = [variable(self.mf/rho + phi0)]
        self.fbcs = []
        self.pbcs = []

        # Material
        if self.params['Material']["material"] == "isotropic exponential form":
            self.material = IsotropicExponentialFormMaterial(self.params['Material'])
        elif self.params['Material']["material"] == "linear poroelastic":
            self.material = LinearPoroelasticMaterial(self.params['Material'])
        elif self.params['Material']["material"] == "Neo-Hookean":
            self.material = NeoHookeanMaterial(self.params['Material'])

        # Set variational forms
        self.MForm, self.dMForm = self.set_fluid_variational_form()


    def create_function_spaces(self):
        V1 = VectorElement('P', self.mesh.ufl_cell(), 1)
        P1 = FiniteElement('P', self.mesh.ufl_cell(), 1)
        P2 = FiniteElement('P', self.mesh.ufl_cell(), 2)
        if self.N == 1:
            FS_M = FunctionSpace(self.mesh, P1)
        FS_F = FunctionSpace(self.mesh, P2)
        FS_V = FunctionSpace(self.mesh, V1)
        return FS_M, FS_F, FS_V


    def add_fluid_dirichlet_condition(self, condition, *args, **kwargs):
        if 'source' in kwargs.keys() and kwargs['source']:
            sub = 0 if self.N > 1 else 0
        else:
            sub = self.N-1
        if 'time' in kwargs.keys() and kwargs['time']:
            self.tconditions.append(condition)
        if self.N == 1:
            self.fbcs.append(DirichletBC(self.FS_M, condition, *args))


    def add_pressure_dirichlet_condition(self, condition, *args, **kwargs):
        if 'source' in kwargs.keys() and kwargs['source']:
            sub = 0 if self.N > 1 else 0
        else:
            sub = self.N-1
        if 'time' in kwargs.keys() and kwargs['time']:
            self.tconditions.append(condition)
        self.pbcs.append(DirichletBC(self.FS_F, condition, *args))


    def sum_fluid_mass(self):
        if self.N == 1:
            return self.mf/self.params['Parameter']['rho']
        else:
            return sum([self.mf[i]
                    for i in range(self.N)])/self.params['Parameter']['rho']


    def set_fluid_variational_form(self):

        m = self.mf
        m_n = self.mf_n

        # Parameters
        self.qi = self.q_in()
        q_out = self.q_out()
        rho = self.rho()
        beta = self.beta()
        k = Constant(1/self.dt())
        dt = Constant(self.dt())
        th, th_ = self.theta()
        n = FacetNormal(self.mesh)

        # theta-rule / Crank-Nicolson
        M = th*m + th_*m_n

        # Fluid variational form
        #A = variable(rho * self.J * inv(self.F) * self.K() * inv(self.F.T))
        A = self.K()
        if self.N == 1:
            vm = TestFunction(self.FS_M)
        # we exclude the erm containing dU-dUn since we are excluding any deformation
        #which means that this term should vanish,
        #dot(grad(M), k*(dU-dU_n))*vm*dx  should go 0
            Form = k*(m - m_n)*vm*dx + inner(-A*grad(self.p[0]), grad(vm))*dx

            # Add inflow terms
            Form += -rho*self.qi*vm*dx

            # Add outflow term
            Form += rho*q_out*vm*dx

        dF = derivative(Form, m, TrialFunction(self.FS_M))

        return Form, dF

    def fluid_solid_coupling(self):
        TOL = self.TOL()
        if self.N == 1:
            FS = self.FS_M
        else:
            FS = self.FS_M.sub(0).collapse()
        for i in range(self.N):
            p = TrialFunction(self.FS_F)
            q = TestFunction(self.FS_F)
            a = p*q*dx
            L = Constant(0.0)
            Ll = (tr(diff(self.Psi, self.F) * self.F.T))/self.phif[i]*q*dx - L*q*dx
            # Ll = (tr(diff(self.Psi, self.F)))*q*dx
            A = assemble(a)
            b = assemble(Ll)
            [bc.apply(A, b) for bc in self.pbcs]
            solver = KrylovSolver('minres', 'hypre_amg')
            prm = solver.parameters
            prm.absolute_tolerance = TOL
            prm.relative_tolerance = TOL*1e3
            prm.maximum_iterations = 1000
            p = Function(self.FS_F)
            solver.solve(A, p.vector(), b)
            self.p[i].assign(project(p, FS))

    def Constitutive_Law(self):

        rho = self.rho()
        phi0 = Constant(self.params['Parameter']['phi'])
        m = self.sum_fluid_mass()
        # Kinematics
        n = FacetNormal(self.mesh)
        #Return the dimension of the space this cell is embedded in
        d = self.mf.geometric_dimension()
        self.I = Identity(d)
        self.F = variable(self.I)
        #self.F = variable(self.I + ufl_grad())
        #self.J = variable(det(self.F))
        one = Constant(1)
        self.J = interpolate(one, self.FS_M)
        self.C = variable(self.F.T*self.F)

        self.Psi = self.material.constitutive_law(J=self.J, C=self.C,
                                                M=m, rho=rho, phi=phi0)
        self.f = project(self.Psi, self.FS_M)
        return self.f


    def choose_solver(self, prob):
        if self.params['Simulation']['solver'] == 'direct':
            return self.direct_solver(prob)
        else:
            return self.iterative_solver(prob)


    def solve(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        tol = self.TOL()
        maxiter = 100
        t = 0.0
        dt = self.dt()

        mprob = NonlinearVariationalProblem(self.MForm, self.mf, bcs=self.fbcs,
                                            J=self.dMForm)
        msol = self.choose_solver(mprob)

        while t < self.params['Parameter']['tf']:

            if mpiRank == 0: utils.print_time(t)

            iter = 0
            eps = 1
            mf_ = Function(self.FS_F)
            while eps > tol and iter < maxiter:
                mf_.assign(self.p[0])
                self.Constitutive_Law()
                self.fluid_solid_coupling()
                msol.solve()
                e = self.p[0] - mf_
                eps = np.sqrt(assemble(e**2*dx))
                iter += 1

            sig = (self.mf-mf_)/Constant(self.dt())-self.rho()*self.q_in()
            sig = project(sig, self.FS_M)
            # Store current solution as previous
            self.mf_n.assign(self.mf)

            m = self.sum_fluid_mass()

            # Kinematics
            #self.Constitutive_Law()




            if self.N ==1:
                yield self.mf, self.Uf, self.p, self.f, t, sig


            t += dt

        # Add a last print so that next output won't overwrite my time print statements
        print()

    def step(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        tol = self.TOL()
        maxiter = 100
        t = 0.0
        dt = self.dt()
        steps =[]

        mprob = NonlinearVariationalProblem(self.MForm, self.mf, bcs=self.fbcs,
                                            J=self.dMForm)
        msol = self.choose_solver(mprob)

        while t < self.params['Parameter']['tf'] and (len(steps) < 1):

            if mpiRank == 0: utils.print_time(t)

            iter = 0
            eps = 1
            mf_ = Function(self.FS_F)
            while eps > tol and iter < maxiter:
                mf_.assign(self.p[0])
                self.Constitutive_Law()
                self.fluid_solid_coupling()
                msol.solve()
                e = self.p[0] - mf_
                eps = np.sqrt(assemble(e**2*dx))
                iter += 1

            sig = (self.mf-mf_)/Constant(self.dt())-self.rho()*self.q_in()
            sig = project(sig, self.FS_M)
            # Store current solution as previous
            self.mf_n.assign(self.mf)

            m = self.sum_fluid_mass()

            # Kinematics
            #self.Constitutive_Law()




            if self.N ==1:
                yield self.mf, self.Uf, self.p, self.f, t, sig

            steps.append(t)
            t += dt

        # Add a last print so that next output won't overwrite my time print statements
        print()


    def direct_solver(self, prob):
        sol = NonlinearVariationalSolver(prob)
        sol.parameters['newton_solver']['linear_solver'] = 'mumps'
        sol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol


    def rho(self):
        return Constant(self.params['Parameter']['rho'])

    def beta(self):
        beta = self.params['Parameter']['beta']
        if isinstance(beta, float):
            beta = [beta]
        return [Constant(b) for b in beta]

    def q_out(self):
        if isinstance(self.params['Parameter']['qo'], str):
            return Expression(self.params['Parameter']['qo'], degree=1)
        else:
            return Constant(self.params['Parameter']['qo'])

    def q_in(self):
        if isinstance(self.params['Parameter']['qi'], str):
            return Expression(self.params['Parameter']['qi'], degree=1)
        else:
            return Constant(self.params['Parameter']['qi'])

    def K(self):
        #if self.N == 1:
        d = self.mf.geometric_dimension()
        I = Identity(d)
        K = Constant(self.params['Parameter']['K'])
        return K*I

    def dt(self):
        return self.params['Parameter']['dt']

    def theta(self):
        theta = self.params['Parameter']['theta']
        return Constant(theta), Constant(1-theta)

    def TOL(self):
        return self.params['Parameter']['TOL']

    def phi(self):
        return Constant(self.params['Parameter']['phi'])

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
fprob = FluidelasticProblem(mesh, params.p)
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

# Dirichlet BC were set according a simulation without boundary conditions.
# For time independency of boundary conditions set kwargs time=False
# no definition of paramtere 'Source' needed since one compartment (N=1)
zero = df.Constant(0.0)
start = df.Constant(2.6)
end = df.Constant(2.6)
fprob.add_fluid_dirichlet_condition(start, boundaries, 1, time=False)
fprob.add_fluid_dirichlet_condition(end, boundaries, 2, time=False)
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
f4 = df.XDMFFile(comm, '../data/{}/psi.xdmf'.format(data_dir))
f5 = df.XDMFFile(comm, '../data/{}/sig.xdmf'.format(data_dir))
# Initialize 'set_xdmf_parameters' for XDMFFiles to be created
#
[set_xdmf_parameters(f1[i]) for i in range(N)]
set_xdmf_parameters(f2)
[set_xdmf_parameters(f3[i]) for i in range(N)]
set_xdmf_parameters(f4)
set_xdmf_parameters(f5)
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
#p_sol = Darcy(mesh)
for Mf, Uf, p, f, t, sig in fprob.step():


    [poro.write_file(f1[i], Uf[i], 'uf{}'.format(i), t) for i in range(N)]
    poro.write_file(f2, Mf, 'mf', t)
    [poro.write_file(f3[i], p[i], 'p{}'.format(i), t) for i in range(N)]
    poro.write_file(f4, f, 'Psi', t)
    poro.write_file(f5, sig, 'sig', t)

    #psi = project(psi, FS_M)
    #File("psi.pvd") << psi
    sum_fluid_mass += df.assemble(Mf*dx)
    # No calculation of theor fluid with qi since qi=0 since only using source term
    #theor_fluid_mass += qi*rho*dt

p_sol = Darcy(mesh)

[f1[i].close() for i in range(N)]
f2.close()
[f3[i].close() for i in range(N)]
f4.close()
f5.close()


#
params.write_config('../data/{}/{}.cfg'.format(data_dir, data_dir))
#
# Finally, the result for the expected sum fluid mass, the calculated sum of the
# fluid mass and the average error over all time steps are ptinted to the screen.
#
#error = errornorm(p, p_sol, 'L2')
print("Sum fluid mass: {}".format(sum_fluid_mass))
#print(error)

#Postprocessing of Darcy-Flow output for visualization
# plot solution
plt.figure()
fig = plot(p_sol)
plt.colorbar(fig)
plt.savefig("darcy.png")
#Postprocessing Velocity
kappa = Constant(1e-7)
phi = Constant(0.3)
elem_v = VectorElement("DG", triangle, 0)
W_v = FunctionSpace(mesh, elem_v)
grad_p = project(grad(p_sol), W_v)
vel_f = - kappa * grad_p / phi
plt.figure()
fig = plot(vel_f)
plt.colorbar(fig)
plt.savefig("vel_f")


elem_viz = VectorElement("CG", triangle, 1)
W_viz = FunctionSpace(mesh, elem_viz)
v_viz = project(vel_f, W_viz)
# Save solution to file in VTK format for paraview
v_file = File('Darcy_vel.pvd')
p_file = File('Darcy_p.pvd')
v_file << v_viz
p_file << p_sol
