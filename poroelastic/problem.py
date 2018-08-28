from dolfin import *

from material_models import *
import utils

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
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

        # Material
        self.material = IsotropicExponentialFormMaterial()

        self.theta = 0.5


    def create_function_spaces(self, mesh):
        El_Y = VectorElement('P', mesh.ufl_cell(), 2)
        El_L = FiniteElement('R', mesh.ufl_cell(), 0)
        El_P = FiniteElement('P', mesh.ufl_cell(), 1)
        El_M = FiniteElement('P', mesh.ufl_cell(), 1)
        El_S = MixedElement([El_Y, El_L])
        FS_S = FunctionSpace(mesh, El_S)
        FS_F = FunctionSpace(mesh, El_M)
        return FS_S, FS_F


    def set_boundary_conditions(self, bcs, domains):
        self.bcs = []
        for bc, domain in zip(bcs, domains):
            self.bcs.append(DirichletBC(self.FS_S.sub(0), bc, domain))


    def sum_fluid_mass(self):
        return 0


    def set_solid_variational_form(self):

        # Test functions
        vy, vl = TestFunctions(self.FS_S)

        # Trial functions
        dU, L = split(self.Us)

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
        rho = self.params.params['rho']
        Psi = self.material.constitutive_law(I1, I2, J, 1.0, rho)
        Psic = Psi + L*(J - 1 - self.sum_fluid_mass()/rho)
        Pi = diff(Psi, variable(E)) + L*J*inv(C)

        Form = inner(F*Pi, grad(vy))*dx + L*vl*dx
        return Form


    def set_fluid_variational_form(self):

        # Test functions
        vm = TestFunction(self.FS_F)

        # Trial functions
        m = self.Uf
        m_n = self.Uf_n

        # Parameters
        rho = Constant(self.params.params['rho'])
        qi = Constant(0.0)
        K = Constant(self.params.params['K'])

        # Solid solution
        dU, L = self.Us.split()
        d = dU.geometric_dimension()
        I = Identity(d)
        F = I + grad(dU)
        J = det(F)
        A = J * inv(F) * K * inv(F.T)
        C = F.T*F
        I1 = J**(-2/3) * tr(C)
        I2 = J**(-4/3) * 0.5 * (tr(C)**2 - tr(C*C))
        Psi = self.material.constitutive_law(I1, I2, J, 1.0, rho)
        p = diff(Psi, variable(J*rho)) - L

        # theta-rule / Crank-Nicolson
        th = Constant(self.theta)
        th_ = Constant(1-self.theta)
        k = Constant(1/self.params.params['dt'])
        M = th*m + th_*m_n

        Form = k*(m - m_n)*vm*dx + dot(dot(grad(M), dU/k), vm)*dx -\
                rho*qi*vm*dx - inner(dot(-A, grad(p)), grad(vm))*dx

        return Form


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

        Ff = self.set_fluid_variational_form()
        Jf = derivative(Ff, self.Uf)

        while t < self.params.params['tf']:

            if mpiRank == 0: utils.print_time(t)

            for x in range(100):

                prob = NonlinearVariationalProblem(Ff, self.Uf, bcs=[], J=Jf,
                        form_compiler_parameters={"optimize": True})
                sol = NonlinearVariationalSolver(prob)
                sol.parameters['newton_solver']['linear_solver'] = 'mumps'
                sol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
                sol.parameters['newton_solver']['krylov_solver']['monitor_convergence'] = True
                sol.solve()

                break

            # Store current solution as previous
            self.Uf_n.assign(self.Uf)

            t += dt

            yield self.Uf, t


    def iterative_solver(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        TOL = self.params.params['TOL']

        t = 0.0
        dt = self.params.params['dt']

        Ff = self.set_fluid_variational_form()
        Jf = derivative(F, self.Uf)

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
