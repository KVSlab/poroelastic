from dolfin import *

set_log_level(30)


def PoroelasticProblem(object):

    def __init__(self, mesh, params):
        self.params = ParamParser(params)

        # Create function spaces
        self.SFS = self.create_function_spaces(mesh)

        # Create solution functions
        self.U = Function(SFS)
        self.U_n = Function(SFS)


    def create_function_spaces(self, mesh):
        El_Y = VectorElement('P', mesh.ufl_cell(), 2)
        El_L = FiniteElement('R', mesh.ufl_cell(), 1)
        El = MixedElement([El_Y, El_L])
        SFS = FunctionSpace(mesh, El)
        return SFS


    def set_boundary_conditions(self, bcs):
        self.bcs = bcs


    def sum_fluid_mass(self):
        return 1


    def set_variational_form(self):

        # Test functions
        vy, vl = TestFunctions(self.YFS)

        # Trial functions
        dU, L = split(self.U)

        # Kinematics
        d = self.U.geometric_dimension()
        I = Identity(d)
        F = I + grad(dU)
        J = det(F)
        C = F.T*F
        E = 0.5 * (C - I)

        # Material definition
        material = IsotropicExponentialFormMaterial()
        rho = self.params.params['rho']
        Psi = material.constitutive_law(1.0, rho)
        Psic = Psi + L*(J - 1 - self.sum_fluid_mass()/rho)
        S = diff(Psi, E) + L*J*inv(C)

        Form = inner(F*S, grad(vy))*dx + (dU*vl + L*vy)*dx
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

            prob = NonlinearVariationalProblem(F, self.u, J=J, form_compiler_parameters={"optimize": True})
            sol = NonlinearVariationalSolver(prob)
            sol.parameters['newton_solver']['linear_solver'] = 'minres'
            sol.parameters['newton_solver']['preconditioner'] = 'jacobi'
            #sol.parameters['newton_solver']['absolute_tolerance'] = TOL
            #sol.parameters['newton_solver']['relative_tolerance'] = TOL
            #sol.parameters['newton_solver']['krylov_solver']['monitor_convergence'] = True
            sol.solve()

            # Store current solution as previous
            self.un.assign(self.U)

            t += dt

            yield self.U, t
