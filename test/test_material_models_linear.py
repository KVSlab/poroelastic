import sys
import re
# sys.path.insert(0, './poroelastic/')
import pytest
import dolfin as df
#from poroelastic import *
from poroelastic.utils import *
from poroelastic.param_parser import *
from poroelastic.material_models import *
import numpy as np
import configparser

def test_init(linearporoelasticmaterial, param_linearporo):
    print(linearporoelasticmaterial.kappa0.values()[0])
    print(param_linearporo["kappa0"])
    assert linearporoelasticmaterial.kappa0.values()[0] == param_linearporo["kappa0"]

@pytest.fixture
def test_param_file_linearporoelastic():
    loc_config_ = '/tmp/param_linearporo.cfg'
    config = open(loc_config_, 'w+')
    config.write(CONFTEST_linear)
    config.close()
    return loc_config_
    #assert config.read() == CONFTEST
# better use tempfile so it is not user set directory?

@pytest.fixture
def param_linearporo(test_param_file_linearporoelastic):
    configure = configparser.ConfigParser()
    #fix default setting of optionxform() function to return lower case
    configure.optionxform = str
    configure.read('/tmp/param_linearporo.cfg')
    section = 'Material'
    options = configure.items(section)
    param_linearporo = {}
    for key, value in options:
        if '*' in value:
            value = re.sub(r" * .*", "", value, flags=re.I)
            param_linearporo[key] = value
        else:
            param_linearporo[key] = value
    return param_linearporo

        #print(key)
        #param_linearporo[key] = value
'''
    param_linearporo["kappa0"] = configure.getfloat('Material','kappa0')
    param_linearporo["kappa1"] = configure.getfloat('kappa1')
    param_linearporo["kappa2"] = configure.getfloat('kappa2')
    param_linearporo["K"] = configure.getfloat('K')
    param_linearporo["M"] = configure.getfloat('M')
    param_linearporo["b"] = configure.getfloat('b')
'''
    #return param_linearporo

@pytest.fixture
def linearporoelasticmaterial(param_linearporo):
    linearporoelasticmaterial = LinearPoroelasticMaterial(param_linearporo)
    return linearporoelasticmaterial




CONFTEST_linear = """\n[Material]
material = "linear poroelastic"
kappa0 = 0.01 * Pa
kappa1 = 2e3 * Pa
kappa2 = 33 * Pa
K = 2.2e5 * Pa
M = 2.18e5 * Pa
b = 1 """
