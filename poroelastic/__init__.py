__version__ = '1.0'
__author__  = 'Alexandra Diem'
__credits__ = ['Alexandra Diem']
__license__ = 'BSD-3'
__maintainer__ = 'Alexandra Diem'
__email__ = 'alexandra@simula.no'

from dolfin import *
import numpy as np

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
