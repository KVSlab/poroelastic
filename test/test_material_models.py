import sys
# sys.path.insert(0, './poroelastic/')
import pytest
import dolfin as df
from poroelastic import *
#from poroelastic.param_parser import *
#from poroelastic.material_models import *
import numpy as np
import configparser

def isnear(a, b, tol=1.e-11, reltol=1.e-10):
    """
    Check near-equality between two floats to a certain tolerance. Name
    contains 'is' to differentiate it from DOLFIN near()-function.
    Arguments
    ---------
    a : float
        First number
    b : float
        Second number
    tol : float
        Tolerance for near-equality
    reltol : float
        Relative tolerance for near-equality
    Returns
    -------
    return : boolean
        True if a and b are near-equal
    """
    # Neglect relative error if numbers are close to zero
    if np.abs(b) > 1.e-10:
        return np.abs(a-b) < tol or np.abs(a/b-1) < reltol
    else:
        return np.abs(a-b) < tol



def test_init(isotropicexponentialformmaterial, param):

    a, D1, D2, D3, Qi1, Qi2, Qi3 = param
    #assert(near(isotropicexponentialformmaterial.a, a))
    #assert(isnear(isotropicexponentialformmaterial.a,a))

"""
def test_init(self):
    #a = param
    att = IsotropicExponentialFormMaterial(self.a)
    assert att == 1.0
def test_no_value():
    with pytest.raises(Exception) as e_info:
        obj = IsotropicExponentialFormMaterial()
#def test_test():
    #print("I am a test to test the test.")
"""

CONFTEST = """\n[Simulation]
sim = sanity_check
solver = direct
debug = 0

[Units]
s = 1
m = 1
Pa = 1
mmHg = 133.322365 * Pa
kPa = 1000 * Pa
kg = 1 * Pa*m*s**2

[Parameter]
N = 1
TOL = 1e-7
rho = 1000 * kg/m**3
K = 1e-7 * m**2/Pa/s
phi = 0.3
beta = 0.2
qi = 10.0 * m**3/s
qo = 0.0 * m**3/s
tf = 1.0 * s
dt = 5e-2 * s
theta = 0.5

[Material]
material = "isotropic exponential form"
a = 1.0
D1 = 2.0
D2 = 0.2
D3 = 2.0
Qi1 = 1.0
Qi2 = 0.5
Qi3 = 1.0"""

@pytest.fixture
def test_param_file():
    loc_config = '/tmp/param.cfg'
    config = open(loc_config, 'w+')
    config.write(CONFTEST)
    config.close()
    return loc_config
    #assert config.read() == CONFTEST
# better use tempfile so it is not user set directory?

@pytest.fixture
def param(test_param_file):
    configure = configparser.ConfigParser()
    configure.read('/tmp/param.cfg')
    param = {}
    #param["a"] dict
    # read in parameters to use later
    #param["a"]= configure['Material']['a']
    '''
    a = configure.getfloat('Material','a')
    D1 = configure.getfloat('Material','D1')
    D2 = configure.getfloat('Material','D2')
    D3 = configure.getfloat('Material','D3')
    Qi1 = configure.getfloat('Material','Qi1')
    Qi2 = configure.getfloat('Material','Qi2')
    Qi3 = configure.getfloat('Material','Qi3')

    return a, D1, D2, D3, Qi1, Qi2, Qi3
'''
    param["a"] = configure.getfloat('Material','a')
    param["D1"] = configure.getfloat('Material','D1')
    param["D2"] = configure.getfloat('Material','D2')
    param["D3"] = configure.getfloat('Material','D3')
    param["Qi1"] = configure.getfloat('Material','Qi1')
    param["Qi2"] = configure.getfloat('Material','Qi2')
    param["Qi3"] = configure.getfloat('Material','Qi3')

    return param


@pytest.fixture
def isotropicexponentialformmaterial(param):
    print(type(param))
    a, D1, D2, D3, Qi1, Qi2, Qi3 = param
    #assign value from 'real' class
    isotropicexponentialformmaterial = IsotropicExponentialFormMaterial(param)
    return isotropicexponentialformmaterial
