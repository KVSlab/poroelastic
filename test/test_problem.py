import dolfin as df
from ufl import grad as ufl_grad
import sys
import numpy as np
import pytest
from poroelastic import *
#import poroelastic.utils as utils
#import fenics as fenics
import configparser



def test_init(poroelasticproblem, params, mesh):
    ''' Test correct assignment of initial values. Test correct type of functions. '''

    assert (poroelasticproblem.N) == section_parameter['Parameter']['N']
    print('1')
    assert(type(poroelasticproblem.Us)) == dolfin.Function
    assert(type(poroelasticproblem.Us_n)) == dolfin.Function
    assert(type(poroelasticproblem.mf)) == dolfin.Function
    assert(type(poroelasticproblem.mf_n)) == dolfin.Function
    '''
def test_sum_fluid_mass(poroelasticproblem, params):
    Calculate sum fluid mass and compare to original function

    if poroelasticproblem.N.values()[0] == 1:
        test_sum_fluid_mass =
    '''
@pytest.fixture
def test_param_file():
    loc_config = '/tmp/params_prob.cfg'
    config = open(loc_config, 'w+')
    config.write(CONFTEST)
    config.close()
    return loc_config
    #assert config.read() == CONFTEST
# better use tempfile so it is not user set directory?

@pytest.fixture
def params(test_param_file):
    configure = configparser.ConfigParser()
    configure.optionxform = str
    configure.read('/tmp/params_prob.cfg')
    section_U = 'Units'
    options = configure.items(section_U)
    units = {}
    for key, value in options:
        value = eval(value, units)
        units[key] = value
    #get material section and check with unit section units, strip accordingly
    section = 'Parameter'
    options_ = configure.items(section)
    section_parameter = {}
    for key, value in options_:
        if "\n" in value:
            #check for new line and strip leading/trailing characters, split accordingly and evalute units
            value = list(filter(None, [x.strip() for x in value.splitlines()]))
            value = [eval(val, units) for val in value]

        else:
            value = eval(value,units)
        section_parameter[key] = value

    return units, section_parameter

@pytest.fixture
def poroelasticproblem(params):
    section_U, section_parameter = params
    nx = 10
    mesh = df.UnitSquareMesh(nx, nx)
    poroelasticproblem = PoroelasticProblem(mesh, section_parameter)
    return poroelasticproblem

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
