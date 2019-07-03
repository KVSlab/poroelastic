from poroelastic.utils import *

def test_periodic():
    assert periodic(5.0, 1.0) == 1.0
    assert periodic(5.0, 2.0) == 1.0
