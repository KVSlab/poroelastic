import sys
# sys.path.insert(0, './poroelastic/')
import pytest
import dolfin as df
from poroelastic.utils import *
from poroelastic.param_parser import *
from poroelastic.material_models import *
import numpy as np

@pytest.fixture
def test_init(material_models, param):
    a = param
    #material_models.IsotropicExponentialFormMaterial(material_models, param)
    assert material_models.IsotropicExponentialFormMaterial.a == 1.0

def test_test():
    print("I am a test to test the test.")
