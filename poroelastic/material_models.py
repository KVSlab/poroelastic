from dolfin import *

set_log_level(30)


class IsotropicExponentialFormMaterial(object):

    # Parameters
    a = 1.0
    D1 = 2.0
    D2 = 0.2
    D3 = 2.0
    Qi1 = 1.0
    Qi2 = 0.5
    Qi3 = 1.0

    def __init__(self):
        pass


    def constitutive_law(self, I1, I2, J, M, rho):
        Psi = self.a * (exp(self.D1 * (I1 * (1 + self.Qi1*M/rho) - 3)\
            + self.D2 * (I2 *(1 + self.Qi2*M/rho) - 3)\
            + self.D3 * ((J-1)**2 + self.Qi3*(M/rho)**2)) - 1)
        return Psi


class NeoHookeanMaterial(object):

    # Parameters
    E = 10.0
    nu = 0.3
    mu = Constant(E/(2*(1 + nu)))
    lm = Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    def __init__(self):
        pass


    def constitutive_law(self, Ic, J):
        Psi = (self.mu/2)*(Ic - 3) - self.mu*ln(J) + (self.lm/2)*(ln(J))**2
        return Psi
