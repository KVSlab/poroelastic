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

    def __init__(self, mesh, params):
        self.mesh = mesh
        self.params = params

        # Create function spaces
        self.FS_S, self.FS_F, self.FS_V = self.create_function_spaces()

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
        material = IsotropicExponentialFormMaterial()
        # material = NeoHookeanMaterial()

        # Set variational forms
        self.SForm, self.dSForm, self.Psi = self.set_solid_variational_form(material)
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


    def add_solid_dirichlet_condition(self, condition, boundary, marker, n=-1,
                                        time=False, **kwargs):
        if n != -1:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0).sub(n), condition,
                                boundary, marker, **kwargs))
        else:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0), condition,
                                boundary, marker, **kwargs))
        if time: self.tconditions.append(condition)


    def add_fluid_dirichlet_condition(self, condition, boundary, marker, **kwargs):
        self.fbcs.append(DirichletBC(self.FS_F, condition, boundary, marker))
        if time: self.tconditions.append(condition)


    def sum_fluid_mass(self):
        return self.mf/self.params.params['rho']


    def set_solid_variational_form(self, material):

        U = self.Us
        dU, L = split(self.Us)

        # parameters
        rho = Constant(self.params.params['rho'])

        # fluid Solution
        m = self.mf

        # Kinematics
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))
        C = variable(F.T*F)
        E = variable(0.5 * (C - I))

        # modified Cauchy-Green invariants
        I1 = variable(J**(-2/3) * tr(C))
        I2 = variable(J**(-4/3) * 0.5 * (tr(C)**2 - tr(C*C)))

        # Material definition
        Psi = material.constitutive_law(I1, I2, J, m, rho)
        # Psi = material.constitutive_law(I1, J)
        Psic = Psi + L*(J-m/rho-Constant(1))

        Form = derivative(Psic*dx, U, TestFunction(self.FS_S))
        dF = derivative(Form, U, TrialFunction(self.FS_S))

        return Form, dF, Psi


    def set_fluid_variational_form(self):

        m = self.mf
        # m = TrialFunction(self.FS_F)
        m_n = self.mf_n
        vm = TestFunction(self.FS_F)
        dU, L = self.Us.split()
        dU_n, L_n = self.Us_n.split()

        # Parameters
        rho = self.rho()
        phi0 = self.phi()
        # qi = Constant(2.3e-2)
        qi = Constant(1e-2)
        Ki = self.K()
        k = Constant(1/self.dt())
        th, th_ = self.theta()

        # Kinematics from solid
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))

        VK = TensorFunctionSpace(self.mesh, "P", 1)
        if d == 2:
            exp = Expression((('0.5', '0.0'),('0.0', '1.0')), degree=1)
        elif d == 3:
            exp = Expression((('0.5', '0.0', '0.0'),('0.0', '1.0', '0.0'),
                                ('0.0', '0.0', '1.0')), degree=1)
        self.K = project(Ki*exp, VK, solver_type='mumps')

        # theta-rule / Crank-Nicolson
        M = th*m + th_*m_n

        # Fluid variational form
        A = variable(rho * J * inv(F) * self.K * inv(F.T))
        Form = k*(m - m_n)*vm*dx + dot(grad(M), k*(dU))*vm*dx -\
                rho*qi*vm*dx - inner(-A*grad(self.p), grad(vm))*dx
        dF = derivative(Form, m, TrialFunction(self.FS_F))

        return Form, dF
        # return Form, None


    def fluid_solid_coupling(self):
        dU, L = self.Us.split()
        rho = self.rho()
        phi0 = self.phi()
        d = dU.geometric_dimension()
        I = Identity(d)
        F = variable(I + ufl_grad(dU))
        J = variable(det(F))
        Jphi = variable(J*(self.mf + rho*phi0)/rho)
        # self.p = project(diff(self.Psi, Jphi) - L, self.FS_F)
        p = Expression("1000 * (1 - exp(-pow(x[0], 2)/0.25))", degree=1)
        self.p = project(p, self.FS_F)


    def calculate_flow_vector(self):
        FS = VectorFunctionSpace(self.mesh, 'P', 1)
        dU, L = self.Us.split()
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
        phi = variable((self.mf + rho*phi0)/(rho*J))

        a = (1/rho)*inner(F*m, mv)*dx
        L = inner(-J*self.K*inv(F.T)*grad(self.p), mv)*dx

        solve(a == L, self.Uf, solver_parameters={"linear_solver": "minres",
                                                "preconditioner": "hypre_amg"})


    def move_mesh(self):
        dU, L = self.Us.split()
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
        # a_M = lhs(self.MForm)
        # L_M = rhs(self.MForm)
        # A_M = assemble(a_M)
        # msol = LUSolver(A_M, "mumps")
        # msol.parameters['reuse_factorization'] = True

        sprob = NonlinearVariationalProblem(self.SForm, self.Us, bcs=self.sbcs,
                                            J=self.dSForm)
        ssol = self.choose_solver(sprob)

        while t < self.params.params['tf']:

            if mpiRank == 0: utils.print_time(t)

            for con in self.tconditions:
                con.t = t

            # b_M = assemble(L_M)
            # for bc in self.fbcs:
            #     bc.apply(b_M)

            iter = 0
            eps = 1
            mf_ = Function(self.FS_F)
            while eps > tol and iter < maxiter:
                iter += 1
                ssol.solve()
                msol.solve()
                # msol.solve(A_M, self.mf.vector(), b_M)
                self.fluid_solid_coupling()
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
        sol.parameters['newton_solver']['preconditioner'] = 'jacobi'
        sol.parameters['newton_solver']['absolute_tolerance'] = TOL
        sol.parameters['newton_solver']['relative_tolerance'] = TOL
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol


    def rho(self):
        return Constant(self.params.params['rho'])

    def phi(self):
        return Constant(self.params.params['phi'])

    def K(self):
        kparam = self.params.params['K']
        if isinstance(kparam, str):
            K = Expression(self.params.params['K'],
                                            element=self.FS_F.ufl_element())
        elif isinstance(kparam, float) or isinstance(kparam, int):
            K = Constant(kparam)
        return K

    def dt(self):
        return self.params.params['dt']

    def theta(self):
        theta = self.params.params['theta']
        return Constant(theta), Constant(1-theta)

    def TOL(self):
        return self.params.params['TOL']
