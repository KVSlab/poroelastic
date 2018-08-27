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
