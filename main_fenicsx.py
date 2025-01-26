from pathlib import Path
from typing import NamedTuple

# import meshio

from mpi4py import MPI
import dolfinx
import pandas as pd
import adios4dolfinx

# import scifem
import ufl
import numpy as np
import basix


from utils import Sex, Case, case_parameters, get_lead_positions


def convert_data(datadir):
    import pyvista as pv

    mesh_file = "ModVent_PAP_hexaVol_0-4mm_17endo-42epi_Labeled_full (1).vtk"
    print(f"Reading mesh file {mesh_file}")
    reader = pv.get_reader(mesh_file)
    vtk_mesh = reader.read()

    datadir.mkdir(exist_ok=True)
    data_file = datadir / "mesh.xdmf"
    print(f"Save mesh to XDMF file {data_file}")
    pv.save_meshio(datadir / "mesh.xdmf", vtk_mesh)

    comm = MPI.COMM_WORLD
    print("Load mesh into dolfinx")
    with dolfinx.io.XDMFFile(comm, data_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    print("Create endo_epi function")
    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    endo_epi = dolfinx.fem.Function(V, name="EndoToEpi")

    endo_epi.x.array[:] = vtk_mesh.point_data.get_array("EndoToEpi")[
        mesh.geometry.input_global_indices
    ]
    print("Save endo_epi function for visualization")
    with dolfinx.io.VTXWriter(
        mesh.comm, datadir / "endo_epi_viz.bp", [endo_epi], engine="BP5"
    ) as vtx:
        vtx.write(0.0)
    print("Save endo_epi function for simulation")
    adios4dolfinx.write_function_on_input_mesh(
        datadir / "endo_epi.bp", endo_epi, time=0.0, name="endo_epi"
    )

    print("Create fiber function")
    W = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    fiber = dolfinx.fem.Function(W, name="Fiber")
    fiber_data = vtk_mesh.cell_data.get_array("FibreOrientation")

    inds = mesh.topology.original_cell_index
    fiber.x.array[::3] = fiber_data[inds, 0]
    fiber.x.array[1::3] = fiber_data[inds, 1]
    fiber.x.array[2::3] = fiber_data[inds, 2]
    print("Save fiber function for visualization")
    with dolfinx.io.VTXWriter(
        mesh.comm, datadir / "fiber_viz.bp", [fiber], engine="BP5"
    ) as vtx:
        vtx.write(0.0)

    print("Save fiber function for simulation")
    adios4dolfinx.write_function_on_input_mesh(
        datadir / "fiber.bp", endo_epi, time=0.0, name="fiber"
    )

    print("Create stimulus")
    create_stimulus(mesh, datadir)


class Geometry(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    fiber: dolfinx.fem.Function
    endo_epi: dolfinx.fem.Function


def load_data(datadir) -> Geometry:
    if not datadir.is_dir():
        convert_data(datadir)
    comm = MPI.COMM_WORLD
    print("Load mesh")
    with dolfinx.io.XDMFFile(comm, datadir / "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    endo_epi = dolfinx.fem.Function(V, name="EndoToEpi")

    print("Load endo_epi function")
    adios4dolfinx.read_function(datadir / "endo_epi.bp", endo_epi, name="endo_epi")

    W = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    fiber = dolfinx.fem.Function(W, name="Fiber")
    print("Load fiber function")
    adios4dolfinx.read_function(datadir / "fiber.bp", fiber, name="fiber")
    return Geometry(mesh, fiber, endo_epi)


def create_stimulus(mesh, datadir):
    print("Create stimulus")
    df = pd.read_csv("tact_pmj.csv")

    W = dolfinx.fem.functionspace(mesh, ("DG", 0))

    Istim = dolfinx.fem.Function(W, name="Istim")
    import shutil

    shutil.rmtree(datadir / "stim_viz.bp", ignore_errors=True)
    shutil.rmtree(datadir / "Istim.bp", ignore_errors=True)
    vtx = dolfinx.io.VTXWriter(
        mesh.comm, datadir / "stim_viz.bp", [Istim], engine="BP5"
    )

    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim, padding=0.01)
    for t in df.Activation.unique():
        print("Create stimulus at time", t)
        points_t = df[df["Activation"] == 1]
        inds = []
        for i, row in points_t.iterrows():
            points = np.array([row.Points_0, row.Points_1, row.Points_2])
            collisions = dolfinx.geometry.compute_collisions_points(tree, points)
            if len(collisions) == 0:
                continue
            cell = collisions.array[0]
            inds.append(cell)

        Istim.x.array[np.array(inds)] = 1.0
        adios4dolfinx.write_function_on_input_mesh(
            datadir / "Istim.bp", Istim, time=t, name="Istim"
        )
        vtx.write(t)
        Istim.x.array[:] = 0.0


def load_stimulus(Istim, datadir, t, stim_duration=2):
    T = int(t)
    if T >= 50:
        Istim.x.array[:] = 0.0
        return
    T_start = max(T - stim_duration, 0)
    T_end = min(T + stim_duration, 50)
    Istim_tmp = dolfinx.fem.Function(Istim.function_space)
    for t in range(T_start, T_end):
        adios4dolfinx.read_function(
            datadir / "Istim.bp", Istim_tmp, name="Istim", time=t
        )
        Istim.x.array += Istim_tmp.x.array


def main():
    datadir = Path("hex-mesh")
    geo = load_data(datadir)

    # if cell is not None:
    # inds.append(cell)
    breakpoint()


if __name__ == "__main__":
    main()
