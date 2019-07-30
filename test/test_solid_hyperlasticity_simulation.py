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
import pdb


# Compiler parameters
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["allow_extrapolation"] = True

set_log_level(0)

# class InitialConditions(UserExpression):
#     def eval(self, value, x):
#         value[0]= Expression((1e-2),degree=1)
#         value[1]=0.0
#     def value_shape(self):
#         return (1,)
class HyperElasticProblem(object):
    """
    Boundary marker labels:
    - inflow (Neumann BC in fluid mass increase)
    - outflow (Neumann BC in fluid mass increase)
    """

    def __init__(self, mesh, params, boundaries=None, markers={}):
        self.mesh = mesh
        self.params = params
        self.markers = markers
        #self.N = int(self.params['Parameter']['N'])

        if boundaries != None:
            self.ds = ds(subdomain_data=boundaries)
        else:
            self.ds = ds()

        # Create function spaces
        self.FS_S = self.create_function_spaces()

        # Create solution functions
        # self.Us = Function(self.FS_S)
        # give initial value for solution of solid form
        # u_init = InitialConditions()
        d = self.mesh.topology().dim()
        initial = Constant([1e-2 for i in range(d)])
        us0 = interpolate(initial, self.FS_S)
        self.Us_n = Function(self.FS_S)
        self.Us = Function(self.FS_S)
        assign(self.Us, us0)

        # initial = interpolate(self.FS, FS_S)
        #self.Us. = interpolate(1e-2, self.FS_S, FS_S)


        phi0 = self.phi()

        self.sbcs = []


        if self.params['Material']["material"] == "Neo-Hookean":
            self.material = NeoHookeanMaterial(self.params['Material'])
        else:
            print("Unable to find right material.")

        # Set variational forms
        self.SForm, self.dSForm, Psic = self.set_solid_variational_form({})

    def create_function_spaces(self):
        V2 = VectorElement('P', self.mesh.ufl_cell(), 2)
        P1 = FiniteElement('P', self.mesh.ufl_cell(), 1)
        TH = MixedElement([V2, P1]) # Taylor-Hood element
        FS_S = FunctionSpace(self.mesh, V2)
        #FS_S = dolfin.FunctionSpace(self.mesh,TH)

        #FS_S = VectorFunctionSpace(self.mesh, 'P', 1)

        return FS_S

    def add_solid_dirichlet_condition(self, condition, *args, **kwargs):
        if 'n' in kwargs.keys():
            n = kwargs['n']
            dkwargs = {}
            # if 'method' in kwargs.keys():
            #     dkwargs['method'] = kwargs['method']
            # self.sbcs.append(DirichletBC(self.FS_S.sub(0).sub(n), condition,
            #                     *args, **dkwargs))
        else:
            self.sbcs.append(DirichletBC(self.FS_S, condition,
                                *args, **kwargs))
        if 'time' in kwargs.keys() and kwargs['time']:
            self.tconditions.append(condition)

    def add_solid_neumann_conditions(self, conditions, boundaries):
        self.SForm, self.dSForm =\
                    self.set_solid_variational_form(zip(conditions, boundaries))

    def set_solid_variational_form(self, neumann_bcs):

        U = self.Us
        dU = U
        V = TestFunction(self.FS_S)
        v = V

        # parameters
        phi0 = Constant(self.params['Parameter']['phi'])
        # Kinematics
        n = FacetNormal(self.mesh)
        #Return the dimension of the space this cell is embedded in
        d = dU.geometric_dimension()
        self.I = Identity(d)
        self.F = variable(self.I + ufl_grad(dU))
        self.J = variable(det(self.F))
        self.C = variable(self.F.T*self.F)

        self.Psi = self.material.constitutive_law(J=self.J, C=self.C,
                                                phi=phi0)
        Psic = self.Psi*dx #+ L*(self.J-Constant(1)-m/rho)*dx

        for condition, boundary in neumann_bcs:
            Psic += dot(condition*n, dU)*self.ds(boundary)

        Form = derivative(Psic, U, V)
        dF = derivative(Form, U, TrialFunction(self.FS_S))

        return Form, dF, Psic

    def move_mesh(self):
        dU = self.Us
        ALE.move(self.mesh, project(dU, VectorFunctionSpace(self.mesh, 'P', 1)))


    def choose_solver(self, prob):
        if self.params['Simulation']['solver'] == 'direct':
            return self.direct_solver(prob)
        else:
            #return self.iterative_solver(prob)
            print("No other option than direct_solver for minimal example.")

    def solve(self):
        #pdb.set_trace()
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)
        tol = self.TOL()
        maxiter = 100
        t = 0.0
        dt = self.dt()
        sprob = NonlinearVariationalProblem(self.SForm, self.Us, bcs=self.sbcs,J=self.dSForm)
        #pdb.set_trace()
        #breakpoint()
        #PetscInfoAllow(PETSC_TRUE)
        ssol = self.choose_solver(sprob)
        while t < self.params['Parameter']['tf']:
            if mpiRank == 0:
                utils.print_time(t)
            iter = 0
            #eps = 1
            ssol.solve()
            self.Us_n.assign(self.Us)
            yield self.Us, t
            self.move_mesh()
            t += dt

        # Add a last print so that next output won't overwrite my time print statements
        print()


    def direct_solver(self, prob):
        sol = NonlinearVariationalSolver(prob)
        sol.parameters['newton_solver']['linear_solver'] = 'mumps'
        sol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol
    # def iterative_solver(self, prob):
    #     TOL = self.TOL()
    #     sol = NonlinearVariationalSolver(prob)
    #     sol.parameters['newton_solver']['linear_solver'] = 'minres'
    #     sol.parameters['newton_solver']['preconditioner'] = 'hypre_amg'
    #     sol.parameters['newton_solver']['absolute_tolerance'] = TOL
    #     sol.parameters['newton_solver']['relative_tolerance'] = TOL*1e3
    #     sol.parameters['newton_solver']['maximum_iterations'] = 1000
    #     return sol

    def phi(self):
        return Constant(self.params['Parameter']['phi'])


    def dt(self):
        return self.params['Parameter']['dt']

    def theta(self):
        theta = self.params['Parameter']['theta']
        return Constant(theta), Constant(1-theta)

    def TOL(self):
        return self.params['Parameter']['TOL']

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
hprob = HyperElasticProblem(mesh, params.p)
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
hprob.add_solid_dirichlet_condition(zero, boundaries, 1)
hprob.add_solid_dirichlet_condition(r, boundaries, 2)
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

for Us, t in hprob.solve():

    dU = Us

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

u = Hyperelastic_Cube(16,12,12)
# u = Hyperelastic_Cube(24,16,16)
error = errornorm(u, dU, 'L2')
print("The error is: {}".format(error))
