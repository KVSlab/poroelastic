# poroelastic
Implementation of the (multicompartment) poroelastic equations.

## Installation
Poroelastic is only offered using Python 3.x.
```
python3 setup.py install
```
installs the package and its dependencies.

## Requirements

Poroelastic requires FEniCS 2017.2.0, upwards compatibility is suspected, but has not been tested. Because poroelastic requires FEniCS we recommend setting up a Docker container
using the Dockerfile
```
docker build --no-cache -t poroelastic:2017.2.0 .
```
To enter the container you can use
```
docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared "poroelastic:2017.2.0"
```
The tag reflects the FEniCS version used to develop the package.

To view the output Paraview 5.x is required.
