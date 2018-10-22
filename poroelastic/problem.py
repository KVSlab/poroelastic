from dolfin import *

from poroelastic.material_models import *
import poroelastic.utils as utils

# Compiler parameters
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

set_log_level(30)


class PoroelasticProblem(object):

    def __init__(self, mesh, params):
        self.params = params

        # Create function spaces
        self.FS_S, self.FS_F = self.create_function_spaces(mesh)

        # Create solution functions
        self.Us = Function(self.FS_S)
        self.Us_n = Function(self.FS_S)
        self.Uf = Function(self.FS_F)
        self.Uf_n = Function(self.FS_F)

        self.theta = 0.5

        self.sbcs = []
        self.tconditions = []

        # Material
        material = IsotropicExponentialFormMaterial()

        # Set variational forms
        self.SForm, self.dSForm, Psi = self.set_solid_variational_form(material)
        self.FForm, self.dFForm = self.set_fluid_variational_form(Psi)


    def create_function_spaces(self, mesh):
        V2 = VectorElement('P', mesh.ufl_cell(), 2)
        R0 = FiniteElement('R', mesh.ufl_cell(), 0)
        P1 = FiniteElement('P', mesh.ufl_cell(), 1)
        TH = MixedElement([V2, P1]) # Taylor-Hood element
        FS_S = FunctionSpace(mesh, TH)
        FS_F = FunctionSpace(mesh, P1)
        return FS_S, FS_F


    def add_solid_dirichlet_condition(self, condition, boundary, n=-1):
        if n != -1:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0).sub(n), condition,
                                boundary))
        else:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0), condition,
                                boundary))


    def add_solid_t_dirichlet_condition(self, condition, boundary, n=-1):
        if n != -1:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0).sub(n), condition,
                                boundary))
            self.tconditions.append(condition)
        else:
            self.sbcs.append(DirichletBC(self.FS_S.sub(0), condition,
                                boundary))
            self.tconditions.append(condition)


    def sum_fluid_mass(self):
        return self.Uf/self.params.params['rho']


    def set_solid_variational_form(self, material):

        U = self.Us
        dU, L = split(self.Us)

        # parameters
        rho = Constant(self.params.params['rho'])

        # fluid Solution
        m = self.Uf

        # Kinematics
        d = dU.geometric_dimension()
        I = Identity(d)
        F = I + grad(dU)
        J = det(F)
        C = F.T*F
        E = 0.5 * (C - I)

        # modified Cauchy-Green invariants
        I1 = J**(-2/3) * tr(C)
        I2 = J**(-4/3) * 0.5 * (tr(C)**2 - tr(C*C))

        # Material definition
        Psi = material.constitutive_law(I1, I2, J, m, rho)
        Psic = Psi*dx + L*(J - Constant(1) - m/rho)*dx

        Form = derivative(Psic, U, TestFunction(self.FS_S))
        dF = derivative(Form, U, TrialFunction(self.FS_S))

        return Form, dF, Psi


    def set_fluid_variational_form(self, Psi):

        m = self.Uf
        m_n = self.Uf_n
        vm = TestFunction(self.FS_F)
        dU, L = split(self.Us)

        # Parameters
        rho = Constant(self.params.params['rho'])
        phi0 = Constant(self.params.params['phi'])
        qi = Constant(0.0)
        Ki = self.K()
        k = Constant(1/self.params.params['dt'])
        th = Constant(self.theta)
        th_ = Constant(1-self.theta)

        # Kinematics from solid
        d = dU.geometric_dimension()
        I = Identity(d)
        F = I + grad(dU)
        J = det(F)
        K = Ki*I

        # Fluid-solid coupling
        phi = (m - rho*phi0)/(rho*J)
        Jphi = variable(J*phi)
        p = diff(Psi, Jphi) - L

        # theta-rule / Crank-Nicolson
        M = th*m + th_*m_n
        Vt_s = variable(dU/k)

        # Fluid variational form
        A = rho * J * inv(F) * K * inv(F.T)
        Form = k*(m - m_n)*vm*dx + dot(grad(M), Vt_s)*vm*dx -\
                rho*qi*vm*dx - inner(dot(-A, grad(p)), grad(vm))*dx

        dF = derivative(Form, m, TrialFunction(self.FS_F))

        return Form, dF


    def solve(self):
        if self.params.sim['solver'] == 'direct':
            return self.direct_solver()
        else:
            return self.iterative_solver()


    def direct_solver(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        TOL = self.params.params['TOL']

        t = 0.0
        dt = self.params.params['dt']

        fprob = NonlinearVariationalProblem(self.FForm, self.Uf, bcs=[],
                                            J=self.dFForm)
        fsol = NonlinearVariationalSolver(fprob)
        fsol.parameters['newton_solver']['linear_solver'] = 'mumps'
        fsol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
        # fsol.parameters['newton_solver']['krylov_solver']['monitor_convergence'] = True


        sprob = NonlinearVariationalProblem(self.SForm, self.Us, bcs=self.sbcs,
                                            J=self.dSForm)
        ssol = NonlinearVariationalSolver(sprob)
        ssol.parameters['newton_solver']['linear_solver'] = 'mumps'
        ssol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
        # ssol.parameters['newton_solver']['krylov_solver']['monitor_convergence'] = True

        while t < self.params.params['tf']:

            if mpiRank == 0: utils.print_time(t)

            for x in range(10):

                fsol.solve()
                ssol.solve()

                break

            # Store current solution as previous
            self.Uf_n.assign(self.Uf)
            self.Us_n.assign(self.Us)

            yield self.Uf, self.Us, t

            t += dt

            for con in self.tconditions:
                con.t = t


    def iterative_solver(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        TOL = self.params.params['TOL']

        t = 0.0
        dt = self.params.params['dt']

        Ff = self.set_fluid_variational_form()
        Jf = derivative(Ff, self.Uf)

        while t < self.params.params['tf']:

            if mpiRank == 0: utils.print_time(t)

            for x in range(100):

                prob = NonlinearVariationalProblem(Ff, self.Uf, bcs=[], J=Jf,
                        form_compiler_parameters={"optimize": True})
                sol = NonlinearVariationalSolver(prob)
                sol.parameters['newton_solver']['linear_solver'] = 'minres'
                sol.parameters['newton_solver']['preconditioner'] = 'jacobi'
                #sol.parameters['newton_solver']['absolute_tolerance'] = TOL
                #sol.parameters['newton_solver']['relative_tolerance'] = TOL
                sol.parameters['newton_solver']['krylov_solver']['monitor_convergence'] = True
                sol.solve()

                # Store current solution as previous
                self.U_n.assign(self.U)

            t += dt

            yield self.U, t


    def K(self):
        kparam = self.params.params['K']
        if isinstance(kparam, str):
            K = Expression(self.params.params['K'],
                                            element=self.FS_F.ufl_element())
        elif isinstance(kparam, float) or isinstance(kparam, int):
            K = Constant(kparam)
        return K
