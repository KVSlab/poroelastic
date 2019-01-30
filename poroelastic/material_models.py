from dolfin import *

set_log_level(30)


class IsotropicExponentialFormMaterial(object):

    def __init__(self, param):
        # Parameters
        self.a = Constant(param["a"])
        self.D1 = Constant(param["D1"])
        self.D2 = Constant(param["D2"])
        self.D3 = Constant(param["D3"])
        self.Qi1 = Constant(param["Qi1"])
        self.Qi2 = Constant(param["Qi2"])
        self.Qi3 = Constant(param["Qi3"])


    def constitutive_law(self, F, **kwargs):
        # kwargs
        M = kwargs["M"]
        rho = kwargs["rho"]

        # kinematic variables
        J = variable(det(F))
        C = variable(F.T*F)
        I1 = variable(J**(-2/3) * tr(C))
        I2 = variable(J**(-4/3) * 0.5 * (tr(C)**2 - tr(C*C)))

        # constitutive law
        Psi = self.a * (exp(self.D1 * (I1 * (1 + self.Qi1*(M/rho)) - 3)\
            + self.D2 * (I2 *(1 + self.Qi2*(M/rho)) - 3)\
            + self.D3 * ((J-1)**2 + self.Qi3*(M/rho)**2)) - 1)
        return Psi


class LinearPoroelastic(object):

    def __init__(self, param):
        # Parameters
        self.kappa0 = Constant(param["kappa0"])
        self.kappa1 = Constant(param["kappa1"])
        self.kappa2 = Constant(param["kappa2"])
        self.K = Constant(param["K"])
        self.M = Constant(param["M"])
        self.b = Constant(param["b"])


    def constitutive_law(self, F, **kwargs):
        # kwargs
        m = kwargs["M"]
        rho = kwargs["rho"]
        phi0 = kwargs["phi0"]

        # kinematic variables
        J = variable(det(F))
        C = variable(F.T*F)
        I1 = variable(tr(C))
        I2 = variable(0.5 * (tr(C)**2 - tr(C*C)))
        I3 = variable(det(C))
        J1 = I1 * I3**(-1/3)
        J2 = I2 * I3**(-2/3)

        if J == 1:
            f = 1
        else:
            f = 2*(J - 1 - ln(J))/(J-1)**2

        Whyp = self.kappa1 * (J1-3) + self.kappa2 * (J2-3) + self.K * (J-1) + self.K*ln(J)
        Psi = Whyp - self.M*self.b*(m/rho)*(J-1)*f + 0.5*self.M*(m/rho)**2*f + self.kappa0 * ln(m/rho + phi0)
        return Psi


class NeoHookeanMaterial(object):

    def __init__(self, param):
        # Parameters
        E = Constant(param["E"])
        nu = Constant(param["nu"])
        self.mu = Constant(E/(2*(1 + nu)))
        self.lm = Constant(E*nu/((1 + nu)*(1 - 2*nu)))


    def constitutive_law(self, F, **kwargs):
        # kinematic variables
        J = variable(det(F))
        C = variable(F.T*F)
        Ic = variable(J**(-2/3) * tr(C))

        Psi = (self.mu/2)*(Ic - 3) - self.mu*ln(J) + (self.lm/2)*(ln(J))**2
        return Psi
