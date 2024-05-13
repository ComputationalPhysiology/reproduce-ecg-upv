# Reproduce ECG

The repository contains code for reproducing the ECG traces using the mesh from UPV.

You need to make sure to have the file `CI2B_DEF12_17endo-42epi_LABELED.vtk` in the root of the repository. 

## Setup environment
You you need legacy FEniCS, and the easiest way to use use the docker image provided at <https://github.com/scientificcomputing/packages>, i.e

```
docker run --name ecg -w /home/shared -v $PWD:/home/shared -it ghcr.io/scientificcomputing/fenics-gmsh:2024-02-19
```

Next you need to install the dependencies
```
python3 -m pip install -r requirements
```

## Run simulation
```
python3 main.py
```