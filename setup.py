from setuptools import setup

setup(
    name='poroelastic',
    version='0.1.0',
    author='Alexandra K. Diem',
    author_email='alexandra@simula.no',
    packages=['poroelastic', 'test'],
    url='https://github.com/akdiem/poroelastic',
    license='LICENSE',
    description='A Python solver for poroelastic models using DOLFIN',
    long_description=open('README.md').read(),
    install_requires=[
        "fenics == 2017.2.0",
        "numpy",
        "h5py"
    ],
)
