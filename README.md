# poroelastic
Implementation of the (multicompartment) poroelastic equations.

## Installation
Poroelastic is only offered using Python 3.x.
```
python3 setup.py
```
installs the package and its dependencies.

Because poroelastic requires FEniCS we recommend setting up a Docker container
using the Dockerfile
```
docker build --no-cache -t "poroelastic:tag"
```
To enter the container you can use
```
docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared poroelastic:tag
```
