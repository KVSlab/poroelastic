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
        self.FS = self.create_function_spaces(mesh)

        # Create solution functions
        self.U = Function(self.FS)
        self.U_n = Function(self.FS)


    def create_function_spaces(self, mesh):
        El_Y = VectorElement('P', mesh.ufl_cell(), 2)
        El_L = FiniteElement('R', mesh.ufl_cell(), 0)
        El = MixedElement([El_Y, El_L])
        SFS = FunctionSpace(mesh, El)
        return SFS


    def set_boundary_conditions(self, bcs):
        self.bcs = bcs


    def sum_fluid_mass(self):
        return 1


    def set_variational_form(self):

        # Test functions
        vy, vl = TestFunctions(self.FS)

        # Trial functions
        dU, L = split(self.U)

        # Kinematics
        d = self.U.geometric_dimension()
        I = Identity(d)
        F = I + grad(dU)
        J = det(F)
        C = F.T*F
        E = 0.5 * (C - I)

        # modified Cauchy-Green invariants
        I1 = J**(-2/3) * tr(C)
        I2 = J**(-4/3) * 0.5 * (tr(C)**2 - tr(dot(C, C)))

        # Material definition
        material = IsotropicExponentialFormMaterial()
        rho = self.params.params['rho']
        Psi = material.constitutive_law(I1, I2, J, 1.0, rho)
        Psic = Psi + L*(J - 1 - self.sum_fluid_mass()/rho)
        S = diff(Psic, variable(E)) + L*J*inv(C)

        Form = inner(F*S, grad(vy))*dx + L*vl*dx
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

            prob = NonlinearVariationalProblem(F, self.U, J=J, form_compiler_parameters={"optimize": True})
            sol = NonlinearVariationalSolver(prob)
            sol.parameters['newton_solver']['linear_solver'] = 'minres'
            sol.parameters['newton_solver']['preconditioner'] = 'jacobi'
            #sol.parameters['newton_solver']['absolute_tolerance'] = TOL
            #sol.parameters['newton_solver']['relative_tolerance'] = TOL
            #sol.parameters['newton_solver']['krylov_solver']['monitor_convergence'] = True
            sol.solve()

            # Store current solution as previous
            self.U_n.assign(self.U)

            t += dt

            yield self.U, t
