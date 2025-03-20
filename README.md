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

## Convert mesh from vtk to dolfinx
```
python3 main_fenicsx.py convert -i ModVent_PAP_hexaVol_0-4mm_17endo-42epi_Labeled_full\ \(1\).vtk -o hex-mesh
```

## Run simulation

```
python3 main_fenicsx.py run -d hex-mesh -o results-male-control --sex male --case control
```
Note that here you might want to run the command using multiple cores, e.g (with 10 cores). 
```
mpirun -n 10 python3 main_fenicsx.py run -d hex-mesh -o results-male-control --sex male --case control
```

To check which value corresponds to which sex you can run
```
python3 main_fenicsx.py list-sexes
```
and to check which value corresponds to which case you can run
```
python3 main_fenicsx.py list-cases
```

## Postprocess results

### Compute ECG
To compute the ECG you can run
```
python3 main_fenicsx.py ecg -d results