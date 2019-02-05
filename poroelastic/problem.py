from dolfin import *
from ufl import grad as ufl_grad
import sys
import numpy as np

from poroelastic.material_models import *
import poroelastic.utils as utils

# Compiler parameters
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

set_log_level(20)


class PoroelasticProblem(object):
    """
    Boundary marker labels:
    - inflow (Neumann BC in fluid mass increase)
    - outflow (Neumann BC in fluid mass increase)
    """

    def __init__(self, mesh, params, boundaries=None, markers={}, fibers=None, territories=None):
        self.mesh = mesh
        self.params = params
        self.markers = markers

        if boundaries != None:
            self.ds = ds(subdomain_data=boundaries)
        else:
            self.ds = ds()

        if territories == None:
            self.territories = MeshFunction("size_t", mesh, mesh.topology().dim())
            self.territories.set_all(0)
        else:
            self.territories = territories

        # Create function spaces
        self.FS_S, self.FS_F, self.FS_V = self.create_function_spaces()

        if fibers != None:
            self.fibers = Function(self.FS_V, fibers)
        else:
            self.fibers = None

        # Create solution functions
        self.Us = Function(self.FS_S)
        self.Us_n = Function(self.FS_S)
        self.mf = Function(self.FS_F)
        self.mf_n = Function(self.FS_F)
        self.Uf = Function(self.FS_V)
        self.p = Function(self.FS_F)

        self.sbcs = []
        self.fbcs = []
        self.tconditions = []

        # Material
        if self.params.material["material"] == "isotropic exponential form":
            self.material = IsotropicExponentialFormMaterial(self.params.material)
        elif self.params.material["material"] == "linear poroelastic":
            self.material = LinearPoroelasticMaterial(self.params.material)
        elif self.params.material["material"] == "Neo-Hookean":
            self.material = NeoHookeanMaterial(self.params.material)

        # Set variational forms
        self.SForm, self.dSForm = self.set_solid_variational_form({})
        self.MForm, self.dMForm = self.set_fluid_variational_form()


    def create_function_spaces(self):
        V1 = VectorElement('P', self.mesh.ufl_cell(), 1)
        V2 = VectorElement('P', self.mesh.ufl_cell(), 2)
        P1 = FiniteElement('P', self.mesh.ufl_cell(), 1)
        P2 = FiniteElement('P', self.mesh.ufl_cell(), 2)
        TH = MixedElement([V2, P1]) # Taylor-Hood element
        FS_S = FunctionSpace(self.mesh, TH)
        FS_F = FunctionSpace(self.mesh, P1)
        FS_V = FunctionSpace(self.mesh, V1)
        return FS_S, FS_F, FS_V


    def add_solid_dirichlet_condition(self, condition, *args, **kwargs):
        if 'n' in kwargs.keys():
            n = kwargs['n']
            dkwargs = {}
            if 'method' in kwargs.keys():
                dkwargs['method'] = kwargs['method']
            self.sbcs.append(DirichletBC(self.FS_S.sub(0).sub(n), condition,
                                *args, **dkwargs))
        else:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0), condition,
                                *args, **kwargs))
        if 'time' in kwargs.keys() and kargs['time']:
            self.tconditions.append(condition)


    def add_solid_neumann_conditions(self, conditions):
        neumann_bcs = conditions
        self.SForm, self.dSForm = self.set_solid_variational_form(neumann_bcs)


    def sum_fluid_mass(self):
        return self.mf/self.params.params['rho']


    def set_solid_variational_form(self, neumann_bcs):

        U = self.Us
        dU, L = split(self.Us)
        V = TestFunction(self.FS_S)
        v, w = split(V)

        # parameters
        rho = Constant(self.params.params['rho'])

        # fluid Solution
        m = self.mf

        # Kinematics
        n = FacetNormal(self.mesh)
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))
        C = variable(F.T*F)

        # modified Cauchy-Green invariants
        I1 = variable(J**(-2/3) * tr(C))
        I2 = variable(J**(-4/3) * 0.5 * (tr(C)**2 - tr(C*C)))

        Psi = self.material.constitutive_law(F, M=self.mf, rho=self.rho(), phi0=self.phi())
        Psic = Psi*dx + L*(J-m/rho-Constant(1))*dx

        for boundary, condition in neumann_bcs.items():
            Psic += dot(condition*n, dU)*self.ds(boundary)

        Form = derivative(Psic, U, V)
        dF = derivative(Form, U, TrialFunction(self.FS_S))

        return Form, dF


    def set_fluid_variational_form(self):

        m = self.mf
        m_n = self.mf_n
        vm = TestFunction(self.FS_F)
        dU, L = self.Us.split(True)
        dU_n, L_n = self.Us_n.split(True)

        # Parameters
        self.qi = self.q_in()
        q_out = self.q_out()
        rho = self.rho()
        si = Constant(0.0)
        k = Constant(1/self.dt())
        th, th_ = self.theta()

        # Kinematics from solid
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))

        # VK = TensorFunctionSpace(self.mesh, "P", 1)
        # if d == 2:
        #     exp = Expression((('0.5', '0.0'),('0.0', '1.0')), degree=1)
        # elif d == 3:
        #     exp = Expression((('1.0', '0.0', '0.0'),('0.0', '1.0', '0.0'),
        #                         ('0.0', '0.0', '1.0')), degree=1)
        # self.K = project(Ki*exp, VK, solver_type='mumps')

        # theta-rule / Crank-Nicolson
        M = th*m + th_*m_n

        # Fluid variational form
        A = variable(rho * J * inv(F) * self.K() * inv(F.T))
        Form = k*(m - m_n)*vm*dx + dot(grad(M), k*(dU-dU_n))*vm*dx -\
                inner(-A*grad(self.p), grad(vm))*dx + rho*si*vm*dx

        # Add inflow terms
        Form += -self.rho()*self.qi*vm*dx

        # Add outflow term
        Form += self.rho()*q_out*vm*dx

        dF = derivative(Form, m, TrialFunction(self.FS_F))

        return Form, dF


    def fluid_solid_coupling(self, t):
        dU, L = self.Us.split(True)
        p = self.p
        q = TestFunction(self.FS_F)
        rho = self.rho()
        phi0 = self.phi()
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))
        Psi = self.material.constitutive_law(F, M=self.mf, rho=self.rho(), phi0=self.phi())
        phi = (self.mf + rho*phi0)
        Jphi = variable(J*phi)
        p = diff(Psi, Jphi) - L
        self.p = project(p, self.FS_F)


    def calculate_flow_vector(self):
        FS = VectorFunctionSpace(self.mesh, 'P', 1)
        dU, L = self.Us.split(True)
        m = TrialFunction(self.FS_V)
        mv = TestFunction(self.FS_V)

        # Parameters
        rho = Constant(self.rho())
        phi0 = self.phi()
        k = Constant(1/self.dt())

        # Kinematics from solid
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))
        phi = (self.mf + rho*phi0)

        a = (1/rho)*inner(F*m, mv)*dx
        L = inner(-J*self.K()*inv(F.T)*grad(self.p), mv)*dx

        solve(a == L, self.Uf, solver_parameters={"linear_solver": "minres",
                                                "preconditioner": "hypre_amg"})



    def move_mesh(self):
        dU, L = self.Us.split(True)
        ALE.move(self.mesh, project(dU, VectorFunctionSpace(self.mesh, 'P', 1)))


    def choose_solver(self, prob):
        if self.params.sim['solver'] == 'direct':
            return self.direct_solver(prob)
        else:
            return self.iterative_solver(prob)


    def solve(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        tol = self.TOL()
        maxiter = 50
        t = 0.0
        dt = self.dt()

        mprob = NonlinearVariationalProblem(self.MForm, self.mf, bcs=self.fbcs,
                                            J=self.dMForm)
        msol = self.choose_solver(mprob)

        sprob = NonlinearVariationalProblem(self.SForm, self.Us, bcs=self.sbcs,
                                            J=self.dSForm)
        ssol = self.choose_solver(sprob)

        while t < self.params.params['tf']:

            if mpiRank == 0: utils.print_time(t)

            for con in self.tconditions:
                con.t = t

            self.qi.t = t

            iter = 0
            eps = 1
            mf_ = Function(self.FS_F)
            while eps > tol and iter < maxiter:
                iter += 1
                ssol.solve()
                self.fluid_solid_coupling(t)
                msol.solve()
                diff = self.mf.vector().get_local() - mf_.vector().get_local()
                eps = np.linalg.norm(diff, ord=np.Inf)
                mf_.assign(self.mf)

            # Store current solution as previous
            self.mf_n.assign(self.mf)
            self.Us_n.assign(self.Us)

            # Calculate fluid vector
            self.calculate_flow_vector()

            yield self.mf, self.Uf, self.p, self.Us, t

            self.move_mesh()

            t += dt


    def direct_solver(self, prob):
        sol = NonlinearVariationalSolver(prob)
        sol.parameters['newton_solver']['linear_solver'] = 'mumps'
        sol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol


    def iterative_solver(self, prob):
        TOL = self.TOL()
        sol = NonlinearVariationalSolver(prob)
        sol.parameters['newton_solver']['linear_solver'] = 'minres'
        sol.parameters['newton_solver']['preconditioner'] = 'hypre_amg'
        sol.parameters['newton_solver']['absolute_tolerance'] = TOL
        sol.parameters['newton_solver']['relative_tolerance'] = TOL
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol


    def rho(self):
        return Constant(self.params.params['rho'])

    def phi(self):
        return Constant(self.params.params['phi'])

    def q_out(self):
        if isinstance(self.params.params['qo'], str):
            return Expression(self.params.params['qo'], degree=1)
        else:
            return Constant(self.params.params['qo'])

    def q_in(self):
        class Qin(Expression):
            def __init__(self, territories, qin, **kwargs):
                self.territories = territories
                self.qin = qin

            def eval_cell(self, values, x, cell):
                t = self.territories[cell.index]
                values[0] = self.qin[t] * (1 - exp(-pow(x[1], 2)/(2*pow(1.5, 2)))/(sqrt(2*pi)*1.5) * exp(-pow(x[2], 2)/(2*pow(1.5, 2)))/(sqrt(2*pi)*1.5))

        qin = self.params.params['qi']
        if not isinstance(qin, list):
            qin = [qin]

        q = Qin(self.territories, qin, degree=0)
        return q

    def K(self):
        # if self.N == 1:
        d = self.mf.geometric_dimension()
        I = Identity(d)
        K = Constant(self.params.params['K'])
        if self.fibers:
            return K*I
        else:
            return K*I
        # else:
        #     d = self.u[0].geometric_dimension()
        #     I = Identity(d)
        #     K = [Constant(k) for k in self.params.params['K']]
        #     if self.fiber:
        #         return [k*self.fiber*I for k in K]
        #     else:
        #         return [k*I for k in K]

    def dt(self):
        return self.params.params['dt']

    def theta(self):
        theta = self.params.params['theta']
        return Constant(theta), Constant(1-theta)

    def TOL(self):
        return self.params.params['TOL']
