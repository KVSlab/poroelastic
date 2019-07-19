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
    section_U, section_material = param_linearporo

    assert linearporoelasticmaterial.kappa0.values()[0] == section_material["kappa0"]
    assert linearporoelasticmaterial.kappa1.values()[0] == section_material["kappa1"]
    assert linearporoelasticmaterial.kappa2.values()[0] == section_material["kappa2"]
    assert linearporoelasticmaterial.K.values()[0] == section_material["K"]
    assert linearporoelasticmaterial.M.values()[0] == section_material["M"]
    assert linearporoelasticmaterial.b.values()[0] == section_material["b"]

@pytest.fixture
def test_param_file_linearporoelastic():
    loc_config_ = '/tmp/param_linearporo.cfg'
    config = open(loc_config_, 'w+')
    config.write(CONFTEST_linear)
    config.close()
    return loc_config_


@pytest.fixture
def param_linearporo(test_param_file_linearporoelastic):
    configure = configparser.SafeConfigParser()
    #fix default setting of optionxform() function to return lower case
    configure.optionxform = str
    configure.read('/tmp/param_linearporo.cfg')
    configure = configparser.ConfigParser()
    #fix default setting of optionxform() function to return lower case
    configure.optionxform = str
    configure.read('/tmp/param_linearporo.cfg')

    #get unit section
    section_U = 'Units'
    options = configure.items(section_U)
    units = {}
    for key, value in options:
        value = eval(value, units)
        units[key] = value
    #get material section and check with unit section units, strip accordingly
    section = 'Material'
    options_ = configure.items(section)
    section_material = {}
    for key, value in options_:
        if "\n" in value:
            #check for new line and strip leading/trailing characters, split accordingly and evalute units
            value = list(filter(None, [x.strip() for x in value.splitlines()]))
            value = [eval(val, units) for val in value]

        else:
            value = eval(value,units)
        section_material[key] = value

    return section_U, section_material


@pytest.fixture
def linearporoelasticmaterial(param_linearporo):
    #split answer from function
    section_U, section_material = param_linearporo
    linearporoelasticmaterial = LinearPoroelasticMaterial(section_material)
    return linearporoelasticmaterial



CONFTEST_linear = """\n[Units]
s = 1
m = 1
Pa = 1
mmHg = 133.322365 * Pa
kPa = 1000 * Pa
kg = 1 * Pa*m*s**2
[Material]
material = "linear poroelastic"
kappa0 = 0.01 * Pa
kappa1 = 2e3 * Pa
kappa2 = 33 * Pa
K = 2.2e5 * Pa
M = 2.18e5 * Pa
b = 1 """
