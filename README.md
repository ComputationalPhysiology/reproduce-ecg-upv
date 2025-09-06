# Reproduce ECG

The repository contains code for reproducing the ECG traces using the mesh from UPV.

You need to make sure to have the file `CI2B_DEF12_17endo-42epi_LABELED.vtk` in the root of the repository. 

## Setup environment and data
The code uses [FEniCSx](https://fenicsproject.org/) and can be run using [Docker](https://www.docker.com/) or [conda](https://docs.conda.io/en/latest/).

### Using conda
To use conda, please create a new environment using the following command
```bash
conda env create -f environment.yml
```
This will create a new environment called `fenicsx-ecg`. You can activate the environment using
```bash
conda activate fenicsx-ecg
```


### Using docker
To use docker, please pull the dolfinx version 0.9 image
```bash
docker pull ghcr.io/fenics/dolfinx/dolfinx:v0.9.0
```
Then you can run the container using
```bash
docker run --name ecg -w /home/shared -v $PWD:/home/shared -it ghcr.io/fenics/dolfinx/dolfinx:v0.9.0
```
Next you need to install the dependencies inside the container
```bash
python3 -m pip install -r requirements.txt
```


### Convert mesh from vtk to dolfinx
First make sure to download the mesh used for the simulations. The mesh is available at [ADD LINK HERE](/#).

Next we need to convert the mesh from vtk to dolfinx format. This can be done using the following command
```
python3 main_fenicsx.py convert -i ModVent_PAP_hexaVol_0-4mm_17endo-42epi_Labeled_full\ \(1\).vtk -o hex-mesh
```
This will create a folder `hex-mesh` containing the mesh and it will also generate the fiber orientations using the Laplace-Dirichlet method using the [fenicsx-ldrb](https://github.com/finsberg/fenicsx-ldrb/tree/main) library.


## Purkinje activation points
The simulations uses a list of activation times generated using a Purkinje network. The points are stored in the file ["tact_pmj.csv"](tact_pmj.csv).

## Run simulation

To run a simulation you can use the following command

```bash
python3 main_fenicsx.py run -d hex-mesh -o results_profile1/male-control --sex male --case CTRL
```
Note that here you might want to run the command using multiple cores, e.g (with 10 cores). 
```bash
mpirun -n 10 python3 main_fenicsx.py run -d hex-mesh -o results_profile1/male-control --sex male --case CTRL
```

## Postprocess results

There are severeal postprocessing steps that can be done.

### Create movies
To create movies from the results you can run
```bash
main_fenicsx.py viz -d hex-mesh -r results_profile_1/male-control 
```
Which will create a movie in the results folder (here `results_profile_1/male-control`).

### Plotting single cell results

To plot single cell results you can run
```bash
python3 plot_all_single_cell.py
```
(see the script for more options).

### Plotting all ECG results
To plot all ECG results you can run
```bash
python3 plot_all_ecg.py
```
(see the script for more options).

### Compute QT intervals

To compute the QT intervals you can run
```bash
python3 qt_interval.py
```
(see the script for more options).



## Submitting jobs to the ex3 cluster
All results from the papers were run on the [ex3 cluster](https://www.ex3.simula.no).

There are a few helper script to do this

* [submit_single_cell.py](submit_single_cell.py): Script to submit single cell simulations to the cluster (these simulations are run in serial - so it might be a idea to run these first before submitting. the full 3D simulations)
* [submit.py](submit.py): Script to submit all the normal and drug jobs to the cluster.
* [submit_TdP.py](submit_TdP.py): Script to submit all the TdP drug jobs to the cluster. These use initial states from the control simulations.
* [submit_create_movie.py](submit_create_movie.py): Script to create movies from the results on the cluster.
* [submit_create_movie_TdP.py](submit_create_movie_TdP.py): Script to create movies from the TdP results on the cluster.
* [rclone.sbatch](rclone.sbatch): Script to copy results from the cluster to google drive using [rclone](https://rclone.org/).




## License
The code is licensed under the MIT license. See the [LICENSE](LICENSE) file for details

## Author
The code here is written by Henrik N. Finsberg (henriknf@simula.no)