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

        self.theta = 0.5


    def create_function_spaces(self, mesh):
        El_Y = VectorElement('P', mesh.ufl_cell(), 2)
        El_L = FiniteElement('R', mesh.ufl_cell(), 0)
        El_P = FiniteElement('P', mesh.ufl_cell(), 1)
        El_M = FiniteElement('P', mesh.ufl_cell(), 1)
        El_S = MixedElement([El_Y, El_L, El_P, El_M])
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
        material = IsotropicExponentialFormMaterial()
        rho = self.params.params['rho']
        Psi = material.constitutive_law(I1, I2, J, 1.0, rho)
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

        # theta-rule / Crank-Nicolson
        th = Constant(self.theta)
        th_ = Constant(1-self.theta)
        P = th*p + th_*p_n
        M = th*m + th_*m_n

        k = Constant(self.params.params['dt'])

        Form = k*(m - m_n)*vm*dx - M*grad(vm)*dx -\
            rho_f*J*inv(F)*K*inv(F.T)*grad(p)*grad(vm)*dx - rho_f*q*vm*ds -\
            (-J*beta_ik * (pi-pk))*vm*dx

        return Form


    def solve(self):
        if self.params.sim['solver'] == 'direct':
            return self.direct_solver()
        else:
            return self.iterative_solver()


    def direct_solver(self):
        pass


    def iterative_solver(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        TOL = self.params.params['TOL']

        t = 0.0
        dt = self.params.params['dt']

        F = self.set_variational_form()
        J = derivative(F, self.U)

        while t < self.params.params['tf']:

            if mpiRank == 0: utils.print_time(t)

            prob = NonlinearVariationalProblem(F, self.U, bcs=self.bcs, J=J,
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
