import sys
# sys.path.insert(0, './poroelastic/')
import pytest
import dolfin as df
from poroelastic import *
#from poroelastic.param_parser import *
#from poroelastic.material_models import *
import numpy as np
import configparser


def test_init(isotropicexponentialformmaterial, param):
    a, D1, D2, D3, Qi1, Qi2, Qi3 = param
    assert isotropicexponentialformmaterial.a.values()[0] == param["a"]
    assert isotropicexponentialformmaterial.D1.values()[0] == param["D1"]
    assert isotropicexponentialformmaterial.D2.values()[0] == param["D2"]
    assert isotropicexponentialformmaterial.D3.values()[0] == param["D3"]
    assert isotropicexponentialformmaterial.Qi1.values()[0] == param["Qi1"]
    assert isotropicexponentialformmaterial.Qi2.values()[0] == param["Qi2"]
    assert isotropicexponentialformmaterial.Qi3.values()[0] == param["Qi3"]

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
    #a, D1, D2, D3, Qi1, Qi2, Qi3 = param
    #assign value from 'real' class
    isotropicexponentialformmaterial = IsotropicExponentialFormMaterial(param)

    return isotropicexponentialformmaterial


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
