from poroelastic.param_parser import *
import poroelastic as poro
import dolfin as df
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
    configure = configparser.ConfigParser()
    configure.read('/tmp/config.cfg')
    configure.sections()
    subdictionaries = []
    #Check for completeness of sectionsxs
    for section_name in configure.sections():
        #section = configure[section_name]
        subdictionaries.append(section_name)
    #figure out if you can make an assert?
    #params = poro.ParamParser()
    #print(p)
    for candidate in subdictionaries:
        print('{:<12}: {}'.format(candidate, configure.has_section(candidate)))
    #testing whether expected keys are present
    #N_key = 'N' in configure['Parameter']
    #for key in configure['Parameter']:
        #if key in configure['Parameter'] == True:
    #testing whether expected keys are present
    #N_key = 'N' in configure['Parameter']
    #for key in configure['Parameter']:
        #if key in configure['Parameter']:
            #print('{:<12}: {}'.format(key, configure.has_option(candidate,key)))
        #else:
            #print('{} is not in \'Parameters\''.format(key))

def test_get_sim_section(test_configure_file):
    configure = configparser.ConfigParser()
    configure.read('/tmp/config.cfg')
    #assert that all values expected for the keys in subdictionary 'Simulation' are true
    assert configure['Simulation']['sim'] == 'sanity_check'
    assert configure['Simulation']['solver'] == 'direct'
    assert configure['Simulation']['debug'] == '0'

def test_get_units_section(test_configure_file):
    configure = configparser.ConfigParser()
    configure.read('/tmp/config.cfg')
    #assert that all values expected for the keys in subdictionary 'Units' are true
    assert configure['Units']['s'] == '1'
    assert configure['Units']['m'] == '1'
    assert configure['Units']['Pa'] == '1'
    assert configure['Units']['mmHg'] == '133.322365 * Pa'
    assert configure['Units']['kPa'] == '1000 * Pa'
    assert configure['Units']['kg'] == '1 * Pa*m*s**2'

def test_get_param_section(test_configure_file):
    configure = configparser.ConfigParser()
    configure.read('/tmp/config.cfg')
    #assert that all values expected for the keys in subdictionary 'Parameter' are true
    assert configure['Parameter']['N'] == '1'
    assert configure['Parameter']['TOL'] == '1e-7'
    assert configure['Parameter']['rho'] == '1000 * kg/m**3'
    assert configure['Parameter']['K'] == '1e-7 * m**2/Pa/s'
    assert configure['Parameter']['phi'] =='0.3'
    assert configure['Parameter']['beta'] == '0.2'
    assert configure['Parameter']['qi'] == '10.0 * m**3/s'
    assert configure['Parameter']['qo'] == '0.0 * m**3/s'
    assert configure['Parameter']['tf'] == '1.0 * s'
    assert configure['Parameter']['dt'] == '5e-2 * s'
    assert configure['Parameter']['theta'] == '0.5'

def test_get_material_section(test_configure_file):
    configure = configparser.ConfigParser()
    configure.read('/tmp/config.cfg')
    #assert that all values expected for the keys in subdictionary 'Material' are true
    assert isinstance(configure['Material']['material'], str)
    assert configure['Material']['material'] == '"isotropic exponential form"'
    assert configure['Material']['a'] == '1.0'
    assert configure['Material']['D1'] == '2.0'
    assert configure['Material']['D2'] == '0.2'
    assert configure['Material']['D3'] == '2.0'
    assert configure['Material']['Qi1'] == '1.0'
    assert configure['Material']['Qi2'] == '0.5'
    assert configure['Material']['Qi3'] == '1.0'

def test_write_config():
    pass

def test_add_data():
    pass
