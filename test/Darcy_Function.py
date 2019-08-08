
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt


#calculate exact solution
p_e = Expression("1 - x[0]*x[0]", degree = 2)



def Darcy(mesh,sig):

    # Create mesh values to iterate over and define function space
    P = FunctionSpace(mesh, 'P', 1) # Pressure so take P

    # define Constant: here Kappa for permeability
    kappa = Constant(1e-7)

    # Define boundary condition
    def left(x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    def right(x, on_boundary):
        return on_boundary and near(x[0], 1.0)

    bcl = DirichletBC(P, Constant(2.6), left)
    bcr = DirichletBC(P, Constant(2.6), right)
    bcs = [bcl, bcr]

    #Define variational problem

    p = TrialFunction(P) #representing unknown u
    q = TestFunction(P) #testfunction - surprise vadsasas-=w
    #f = Constant(-3.8e-5) # or S for source term
    #self.mf, self.Uf, self.p, self.f, t, sig = FluidelasticProblem.step()
    #f = sig
    a = kappa*dot(grad(p), grad(q))*dx #lhs - left-hand-side term
    L = sig*q*dx #rhs - right-hand-side term

    # Compute solution
    p_sol = Function(P)
    solve(a == L, p_sol, bcs)

    return p_sol
