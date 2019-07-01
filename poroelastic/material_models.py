from dolfin import *
import numpy as np

set_log_level(30)


class IsotropicExponentialFormMaterial(object):
# Config file parameters:
# material = "isotropic exponential form"
# a = 1.0
# D1 = 2.0
# D2 = 0.2
# D3 = 2.0
# Qi1 = 1.0
# Qi2 = 0.5
# Qi3 = 1.0

    def __init__(self, param):
        # Parameters
        self.a = Constant(param["a"])
        self.D1 = Constant(param["D1"])
        self.D2 = Constant(param["D2"])
        self.D3 = Constant(param["D3"])
        self.Qi1 = Constant(param["Qi1"])
        self.Qi2 = Constant(param["Qi2"])
        self.Qi3 = Constant(param["Qi3"])


    def constitutive_law(self, **kwargs):
        # kwargs
        J = kwargs["J"]
        C = kwargs["C"]
        M = kwargs["M"]

        I1 = variable(J**(-2/3) * tr(C))
        I2 = variable(J**(-4/3) * 0.5 * (tr(C)**2 - tr(C*C)))

        # constitutive law
        Psi = self.a * (exp(self.D1 * (I1 * (1 + self.Qi1*M) - 3)\
            + self.D2 * (I2 *(1 + self.Qi2*M) - 3)\
            + self.D3 * ((J-1)**2 + self.Qi3*(M)**2)) - 1)
        return Psi


class LinearPoroelasticMaterial(object):
# Config file parameters:
# material = "linear poroelastic"
# kappa0 = 0.01 * Pa
# kappa1 = 2e3 * Pa
# kappa2 = 33 * Pa
# K = 2.2e5 * Pa
# M = 2.18e5 * Pa
# b = 1

    def __init__(self, param):
        # Parameters
        self.kappa0 = Constant(param["kappa0"])
        self.kappa1 = Constant(param["kappa1"])
        self.kappa2 = Constant(param["kappa2"])
        self.K = Constant(param["K"])
        self.M = Constant(param["M"])
        self.b = Constant(param["b"])


    def constitutive_law(self, **kwargs):
        # kwargs
        M = kwargs["M"]
        phi0 = kwargs["phi"]

        # kinematic variables
        J = kwargs["J"]
        C = kwargs["C"]
        I1 = variable(tr(C))
        I2 = variable(0.5 * (tr(C)**2 - tr(C*C)))
        I3 = variable(det(C))
        J1 = I1 * I3**(-1/3)
        J2 = I2 * I3**(-2/3)

        f = 2*(J - 1 - ln(J))/(J-1)**2
        if np.isnan(assemble(f*dx)):
            f = 1.0

        Whyp = self.kappa1*(J1-3) + self.kappa2*(J2-3) + self.K*(J-1) - self.K*ln(J)
        Psi = Whyp - self.M*self.b*M*(J-1)*f + 0.5*self.M*f*M**2 - self.kappa0*ln(M + phi0)
        return Psi


    def pore_pressure(self, **kwargs):
        J = kwargs["J"]
        M = kwargs["M"]
        fJ = 2*(J-1-ln(J))/(J-1)**2
        if np.isnan(assemble(fJ*dx)):
            fJ = 1
        return self.M * fJ * (self.b*(1-J) + M)


class NeoHookeanMaterial(object):
    # Config file parameters:
    # material = "Neo-Hookean"
    # E = 10.0
    # nu = 0.3

    def __init__(self, param):
        # Parameters
        E = Constant(param["E"])
        nu = Constant(param["nu"])
        self.mu = Constant(E/(2*(1 + nu)))
        self.lm = Constant(E*nu/((1 + nu)*(1 - 2*nu)))


    def constitutive_law(self, **kwargs):
        J = kwargs["J"]
        C = kwargs["C"]
        Ic = tr(C)

        Psi = (self.mu/2)*(Ic - 3) - self.mu*ln(J) + (self.lm/2)*(ln(J))**2
        return Psi
