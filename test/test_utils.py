from poroelastic.utils import *
import numpy as np

def test_periodic():
    t = 5
    T = 1
    a = list(range(0,t,T))
    if a[0] > abs(a[1]-a[0]):
        c = a[0]
    else:
        c = a[1]
    assert periodic(t,T) == c
