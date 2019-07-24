
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt

#calculate exact solution
p_e = Expression("1 - x[0]*x[0]", degree = 2)



def darcy(n):

    # Create mesh values to iterate over and define function space
    mesh = UnitSquareMesh(n, n)
    P = FunctionSpace(mesh, 'P', 1) # Pressure so take P
    #calculate exact solution
    #p_e = Expression("1 - x[0]*x[0]", degree = 2)


    # define Constant: here Kappa for permeability
    kappa = Constant(1e-7)
    #define phi
    #phi = Constant(0.3)

    # Define boundary condition
    def left(x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    def right(x, on_boundary):
        return on_boundary and near(x[0], 1.0)

    bcl = DirichletBC(P, Constant(1e-3), left)
    bcr = DirichletBC(P, Constant(0.88), right)
    bcs = [bcl, bcr]

    #Define variational problem

    p = TrialFunction(P) #representing unknown u
    q = TestFunction(P) #testfunction - surprise v
    f = Constant(2.0) # or S for source term
    a = kappa*dot(grad(p), grad(q))*dx #lhs - left-hand-side term
    L = f*q*dx #rhs - right-hand-side term

    # Compute solution
    p_sol = Function(P)
    solve(a == L, p_sol, bcs)

    #return for count!
    return mesh, p_sol


#let run over different meshes!
#meshes = [2**i for i in range(10)]
error = []
#for n in meshes:
    #all darcy function
mesh, p_sol = darcy(20)
error.append(errornorm(p_e, p_sol, 'L2'))

print(error)
#plt.semilogy(meshes, error)
#plt.savefig("meshes_errors.png")


#plot solution
plot(p_sol)
plt.savefig("p_sol_8.png")
plot(mesh)

plt.figure()
fig = plot(p_sol)
plt.colorbar(fig)
plt.savefig("mesh_8.png")
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
