import os
import sys
import subprocess
import string
import platform

from setuptools import setup, find_packages, Command

# Version number
major = 0
minor = 1

on_rtd = os.environ.get('READTHEDOCS') == 'True'

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python3 "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)


if on_rtd:
    REQUIREMENTS = []
else:
    REQUIREMENTS = [
        "numpy",
        "scipy",
        "matplotlib",
        "dolfin",
    ]


setup(name = "poroelasticity",
      version = "{0}.{1}".format(major, minor),
      description = """
      A poroelasticity solver using Fenics/Dolfin.
      """,
      url='https://github.com/akdiem/poroelasticity',
      author = "Alexandra Diem",
      author_email = "alexandra@simula.no",
      license="BSD version 3",
      install_requires=REQUIREMENTS,
      packages = ["poroelasticity"],
      package_dir = {"poroelasticity": "poroelasticity"},
      )
