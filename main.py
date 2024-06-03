from enum import IntEnum
from typing import NamedTuple
from pathlib import Path
import dolfin

import json
import meshio
import matplotlib.pyplot as plt
import pandas as pd
import beat
import beat.single_cell
import gotranx
from beat.units import ureg

here = Path(__file__).parent


class Geometry(NamedTuple):
    mesh: dolfin.Mesh
    fiber: dolfin.Function
    endo_epi: dolfin.Function
    ffun: dolfin.MeshFunction
    markers: dict[str, int]


def convert_data():
    mesh_file = "CI2B_DEF12_17endo-42epi_LABELED.vtk"
    vtk_mesh = meshio.read(mesh_file)
    data_file = "data.xdmf"
    if not Path(data_file).is_file():
        vtk_mesh.write("data.xdmf")

    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(data_file) as infile:
        infile.read(mesh)

    V = dolfin.FunctionSpace(mesh, "CG", 1)

    W = dolfin.VectorFunctionSpace(mesh, "DG", 0)

    endo_epi = dolfin.Function(V)
    endo_epi.set_allow_extrapolation(True)

    endo_epi.vector().set_local(
        vtk_mesh.point_data["EndoToEpi"][dolfin.dof_to_vertex_map(V), 0]
    )
    with dolfin.XDMFFile("endo_epi.xdmf") as xdmf:
        xdmf.write_checkpoint(
            endo_epi, "endo_epi", 0.0, dolfin.XDMFFile.Encoding.HDF5, False
        )

    ffun = dolfin.MeshFunction("size_t", mesh, 2)

    for facet in dolfin.facets(mesh):
        ffun[facet] = round(endo_epi(facet.midpoint()))

    with dolfin.XDMFFile("ffun.xdmf") as xdmf:
        xdmf.write(ffun)

    fiber_data = vtk_mesh.cell_data["FibreOrientation"][0]
    fiber = dolfin.Function(W)
    fiber_array = fiber.vector().get_local()
    fiber_array[::3] = fiber_data[:, 0]
    fiber_array[1::3] = fiber_data[:, 1]
    fiber_array[2::3] = fiber_data[:, 2]
    fiber.vector().set_local(fiber_array)
    with dolfin.XDMFFile("fiber.xdmf") as xdmf:
        xdmf.write_checkpoint(fiber, "fiber", 0.0, dolfin.XDMFFile.Encoding.HDF5, False)


def get_lead_positions() -> dict[str, tuple[float, float, float]]:
    def name2lead(name):
        if "LA" in name:
            return "LA"
        if "RA" in name:
            return "RA"
        if "LL" in name:
            return "LL"
        raise ValueError(f"Unknown lead name: {name}")

    text = Path("limb_position.txt").read_text()
    leads = {}
    for line in text.splitlines():
        pos, name = line.strip().split("!")
        leads[name2lead(name)] = tuple(float(p) for p in pos.strip().split(" "))

    return leads


def load_data() -> Geometry:
    files = ["data.xdmf", "fiber.xdmf", "endo_epi.xdmf", "ffun.xdmf"]
    if not all(Path(file).is_file() for file in files):
        breakpoint()
        convert_data()

    mesh = dolfin.Mesh()
    with dolfin.XDMFFile("data.xdmf") as infile:
        infile.read(mesh)

    V = dolfin.FunctionSpace(mesh, "CG", 1)
    W = dolfin.VectorFunctionSpace(mesh, "DG", 0)
    endo_epi = dolfin.Function(V)
    with dolfin.XDMFFile("endo_epi.xdmf") as xdmf:
        xdmf.read_checkpoint(endo_epi, "endo_epi", 0)
    fiber = dolfin.Function(W)
    with dolfin.XDMFFile("fiber.xdmf") as xdmf:
        xdmf.read_checkpoint(fiber, "fiber", 0)

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    with dolfin.XDMFFile("ffun.xdmf") as xdmf:
        xdmf.read(ffun)

    markers = {
        "ENDO": 0,
        "EPI": 2,
    }

    return Geometry(
        mesh=mesh, fiber=fiber, endo_epi=endo_epi, ffun=ffun, markers=markers
    )


def save_ecg(
    t, V: dolfin.Function, ecg_file: Path, leads: dict[str, tuple[float, float, float]]
):
    if ecg_file.is_file():
        data = json.loads(ecg_file.read_text())
    else:
        data = []

    mesh = V.function_space().mesh()

    LA = beat.ecg.ecg_recovery(v=V, mesh=mesh, sigma_b=1.0, point=leads["LA"])
    RA = beat.ecg.ecg_recovery(v=V, mesh=mesh, sigma_b=1.0, point=leads["RA"])
    LL = beat.ecg.ecg_recovery(v=V, mesh=mesh, sigma_b=1.0, point=leads["LL"])
    I = LA - RA
    II = LL - RA
    III = LL - LA

    data.append(
        {
            "time": t,
            "I": I,
            "II": II,
            "III": III,
            "LA": LA,
            "RA": RA,
            "LL": LL,
        }
    )
    ecg_file.write_text(json.dumps(data, indent=2))


def plot_ecg():
    outdir = Path("results-male")
    data = json.loads((outdir / "extracellular_potential.json").read_text())
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(df["time"].to_numpy(), df["I"].to_numpy())
    ax[1].plot(df["time"].to_numpy(), df["II"].to_numpy())
    ax[2].plot(df["time"].to_numpy(), df["III"].to_numpy())
    ax[0].set_title("I")
    ax[1].set_title("II")
    ax[2].set_title("III")
    fig.savefig("ecg.png")


class Sex(IntEnum):
    undefined = 0
    male = 1
    female = 2


def main():

    sex = Sex.male

    outdir = Path(f"results-{sex.name}")
    outdir.mkdir(exist_ok=True)

    data = load_data()
    ode = gotranx.load_ode(here / "ORdmm_Land.ode")
    code = gotranx.cli.gotran2py.get_code(
        ode, scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen]
    )
    model = {}
    exec(code, model)

    mesh_unit = "mm"
    V = dolfin.FunctionSpace(data.mesh, "Lagrange", 1)

    init_states = {
        0: beat.single_cell.get_steady_state(
            fun=model["forward_generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](celltype=0, sex=sex.value),
            outdir=outdir / "steady-states-0D" / "mid",
            BCL=1000,
            nbeats=200,
            track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
            dt=0.05,
        ),
        1: beat.single_cell.get_steady_state(
            fun=model["forward_generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](celltype=2, sex=sex.value),
            outdir=outdir / "steady-states-0D" / "endo",
            BCL=1000,
            nbeats=200,
            track_indices=[
                model["state_index"]("v"),
                model["state_index"]("cai"),
                model["state_index"]("nai"),
            ],
            dt=0.05,
        ),
        2: beat.single_cell.get_steady_state(
            fun=model["forward_generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](celltype=1, sex=sex.value),
            outdir=outdir / "steady-states-0D" / "epi",
            BCL=1000,
            nbeats=200,
            track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
            dt=0.05,
        ),
    }

    init_states = {
        0: model["init_state_values"](),
        1: model["init_state_values"](),
        2: model["init_state_values"](),
    }
    # endo = 0, epi = 1, M = 2
    parameters = {
        0: model["init_parameter_values"](amp=0.0, celltype=0, sex=sex.value),
        1: model["init_parameter_values"](amp=0.0, celltype=2, sex=sex.value),
        2: model["init_parameter_values"](amp=0.0, celltype=1, sex=sex.value),
    }
    fun = {
        0: model["forward_generalized_rush_larsen"],
        1: model["forward_generalized_rush_larsen"],
        2: model["forward_generalized_rush_larsen"],
    }
    v_index = {
        0: model["state_index"]("v"),
        1: model["state_index"]("v"),
        2: model["state_index"]("v"),
    }

    # Surface to volume ratio
    chi = 140.0 * ureg("mm**-1")
    # Membrane capacitance
    C_m = 0.01 * ureg("uF/mm**2")

    time = dolfin.Constant(0.0)
    subdomain_data = dolfin.MeshFunction("size_t", data.mesh, 2)
    subdomain_data.set_all(0)
    marker = 1
    subdomain_data.array()[data.ffun.array() == data.markers["ENDO"]] = marker
    I_s = beat.stimulation.define_stimulus(
        mesh=data.mesh,
        chi=chi,
        mesh_unit=mesh_unit,
        time=time,
        subdomain_data=subdomain_data,
        marker=marker,
    )

    M = beat.conductivities.define_conductivity_tensor(chi=chi, f0=data.fiber)

    params = {"preconditioner": "sor", "use_custom_preconditioner": False}
    pde = beat.MonodomainModel(
        time=time,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        mesh=data.mesh,
        M=M,
        I_s=I_s,
        params=params,
    )

    ode = beat.odesolver.DolfinMultiODESolver(
        v_ode=dolfin.Function(V),
        v_pde=pde.state,
        markers=data.endo_epi,
        num_states={i: len(s) for i, s in init_states.items()},
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        v_index=v_index,
    )

    T = 500
    # Change to 500 to simulate the full cardiac cycle
    # T = 500
    t = 0.0
    dt = 0.05
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

    fname = outdir / "state.xdmf"
    if fname.is_file():
        fname.unlink()
        fname.with_suffix(".h5").unlink()

    leads = get_lead_positions()

    ecg_file = outdir / "extracellular_potential.json"

    i = 0
    while t < T + 1e-12:
        if i % 20 == 0:
            v = solver.pde.state.vector().get_local()
            print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() = }")
            with dolfin.XDMFFile(dolfin.MPI.comm_world, fname.as_posix()) as xdmf:
                xdmf.write_checkpoint(
                    solver.pde.state,
                    "V",
                    float(t),
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )
            save_ecg(t, solver.pde.state, ecg_file, leads)

        solver.step((t, t + dt))
        i += 1
        t += dt


if __name__ == "__main__":
    main()
