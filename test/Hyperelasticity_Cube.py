import matplotlib.pyplot as plt
from dolfin import *

def Hyperelastic_Cube(nx, ny, nz):

    # Create mesh and define function space
    mesh = UnitCubeMesh(nx, ny,nz)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    # Mark boundary subdomians
    left = CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
    # .. index:: compiled expression
    #
    # The Dirichlet boundary values are defined using compiled expressions::

    # Define Dirichlet boundary (x = 0 or x = 1)
    c = Constant((0.0, 0.0, 0.0))
    r = Expression(("scale*0.0",
                    "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                    "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                    scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/7, degree=2)

    # Note the use of setting named parameters in the :py:class:`Expression
    # <dolfin.functions.expression.Expression>` for ``r``.
    #
    # The boundary subdomains and the boundary condition expressions are
    # collected together in two :py:class:`DirichletBC
    # <dolfin.fem.bcs.DirichletBC>` objects, one for each part of the
    # Dirichlet boundary::

    bcl = DirichletBC(V, c, left)
    bcr = DirichletBC(V, r, right)
    bcs = [bcl, bcr]

    # The Dirichlet (essential) boundary conditions are constraints on the
    # function space :math:`V`. The function space is therefore required as
    # an argument to :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>`.
    # Define functions
    du = TrialFunction(V)            # Incremental displacement
    v  = TestFunction(V)             # Test function
    u  = Function(V)                 # Displacement from previous iteration

    # Kinematics
    d = len(u)
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Next, the material parameters are set and the strain energy density
    # and the total potential energy are defined, again using UFL syntax::

    # Elasticity parameters
    E, nu = 10.0, 0.3
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx

    return u
