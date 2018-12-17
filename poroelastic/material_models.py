from dolfin import *

set_log_level(30)


class IsotropicExponentialFormMaterial(object):

    def __init__(self):
        # Parameters
        self.a = Constant(1.0)
        self.D1 = Constant(2.0)
        self.D2 = Constant(0.2)
        self.D3 = Constant(2.0)
        self.Qi1 = Constant(1.0)
        self.Qi2 = Constant(0.5)
        self.Qi3 = Constant(1.0)


    def constitutive_law(self, I1, I2, J, M, rho):
        Psi = self.a * (exp(self.D1 * (I1 * (1 + self.Qi1*(M/rho)) - 3)\
            + self.D2 * (I2 *(1 + self.Qi2*(M/rho)) - 3)\
            + self.D3 * ((J-1)**2 + self.Qi3*(M/rho)**2)) - 1)
        return Psi


class NeoHookeanMaterial(object):

    def __init__(self):
        # Parameters
        E = 10.0
        nu = 0.3
        self.mu = Constant(E/(2*(1 + nu)))
        self.lm = Constant(E*nu/((1 + nu)*(1 - 2*nu)))


    def constitutive_law(self, Ic, J):
        Psi = (self.mu/2)*(Ic - 3) - self.mu*ln(J) + (self.lm/2)*(ln(J))**2
        return Psi
