from poroelastic.param_parser import *
import numpy as np
import pytest
import os
import configparser

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
def test_configure_file():
    loc_config = '/tmp/config.cfg'
    config = open(loc_config, 'w+')
    config.write(CONFTEST)
    config.close()
    return loc_config
    #assert config.read() == CONFTEST
# better use tempfile so it is not user set directory?

def test_get_params(test_configure_file):
    """
    Creating a dictionary of paramters by reading the configuration file
    """
    #section = ()

    configure = configparser.ConfigParser()
    configure.read('/tmp/config.cfg')
    configure.sections()
    for section_name in configure.sections():
        #section = configure[section_name]
        section = (section_name)
    #ParamParser.get_params('/tmp/config.cfg')



    #assert section == p
