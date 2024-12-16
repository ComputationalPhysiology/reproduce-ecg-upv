from enum import IntEnum
from typing import NamedTuple
from pathlib import Path
import dolfin
import ufl_legacy as ufl
import json
import meshio
import numba
import numpy as np

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


def load_data(comm) -> Geometry:
    files = ["data.xdmf", "fiber.xdmf", "endo_epi.xdmf", "ffun.xdmf"]
    if not all(Path(file).is_file() for file in files):
        convert_data()
    print("Loading data")
    print("Read mesh")
    mesh = dolfin.Mesh(comm)
    with dolfin.XDMFFile(comm, "data.xdmf") as infile:
        infile.read(mesh)

    print("Read endo-epi")
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    W = dolfin.VectorFunctionSpace(mesh, "DG", 0)
    endo_epi = dolfin.Function(V)
    with dolfin.XDMFFile(comm, "endo_epi.xdmf") as xdmf:
        xdmf.read_checkpoint(endo_epi, "endo_epi", 0)

    print("Read fibers")
    fiber = dolfin.Function(W)
    with dolfin.XDMFFile(comm, "fiber.xdmf") as xdmf:
        xdmf.read_checkpoint(fiber, "fiber", 0)

    print("Read facet function")
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    with dolfin.XDMFFile(comm, "ffun.xdmf") as xdmf:
        xdmf.read(ffun)

    markers = {
        "ENDO": 0,
        "EPI": 2,
    }

    return Geometry(
        mesh=mesh, fiber=fiber, endo_epi=endo_epi, ffun=ffun, markers=markers
    )


def save_ecg(
    comm,
    t,
    V: dolfin.Function,
    ecg_file: Path,
    leads: dict[str, tuple[float, float, float]],
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
    if comm.rank == 0:
        ecg_file.write_text(json.dumps(data, indent=2))


class Sex(IntEnum):
    undefined = 0
    male = 1
    female = 2


class Case(IntEnum):
    control = 0
    dofe = 1


def case_parameters(case: Case) -> dict[str, float]:
    if case == Case.control:
        return {
            "scale_drug_INa": 1.0,
            "scale_drug_INaL": 1.0,
            "scale_drug_Ito": 1.0,
            "scale_drug_ICaL": 1.0,
            "scale_drug_IKr": 1.0,
            "scale_drug_IKs": 1.0,
            "scale_drug_IK1": 1.0,
        }
    elif case == Case.dofe:
        return {
            "scale_drug_INa": 1.0,
            "scale_drug_INaL": 1.0,
            "scale_drug_Ito": 0.75,
            "scale_drug_ICaL": 1.0,
            "scale_drug_IKr": 0.627,
            "scale_drug_IKs": 1.0,
            "scale_drug_IK1": 1.0,
        }
    else:
        raise ValueError(f"Unknown case {case}")


def main(sex: Sex = Sex.male, case: Case = Case.control):
    outdir = Path(f"results-{sex.name}-{case.name}")
    outdir.mkdir(exist_ok=True)

    comm = dolfin.MPI.comm_world
    data = load_data(comm)
    case_ps = case_parameters(case)
    module_path = Path("ORdmm_Land.py")
    if not module_path.is_file():
        ode = gotranx.load_ode(here / "ORdmm_Land.ode")
        code = gotranx.cli.gotran2py.get_code(
            ode, scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen]
        )
        if comm.rank == 0:
            module_path.write_text(code)

    comm.barrier()

    import ORdmm_Land

    model = ORdmm_Land.__dict__

    mesh_unit = "mm"
    V = dolfin.FunctionSpace(data.mesh, "Lagrange", 1)

    celldir = Path(f"results-{sex.name}-control") / "steady-states-0D"

    init_states = {
        0: beat.single_cell.get_steady_state(
            fun=model["forward_generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](celltype=0, sex=sex.value),
            outdir=celldir / "mid",
            BCL=1000,
            nbeats=500,
            track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
            dt=0.05,
        ),
        1: beat.single_cell.get_steady_state(
            fun=model["forward_generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](celltype=2, sex=sex.value),
            outdir=celldir / "endo",
            BCL=1000,
            nbeats=500,
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
            outdir=celldir / "epi",
            BCL=1000,
            nbeats=500,
            track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
            dt=0.05,
        ),
    }

    # else:
    #     # Use initial states from control
    #     init_states = {}
    #     init_states_dir = Path(f"results-{sex.name}-control")
    #     for marker in [0, 1, 2]:
    #         state_name = init_states_dir / f"state_{marker}_{comm.rank}_{comm.size}.npy"

    #         if not state_name.is_file():
    #             raise RuntimeError(f"Missing initial states file {state_name}")
    #         init_states[marker] = np.load(state_name)

    # endo = 0, epi = 1, M = 2
    parameters = {
        0: model["init_parameter_values"](
            amp=0.0, celltype=0, sex=sex.value, **case_ps
        ),
        1: model["init_parameter_values"](
            amp=0.0, celltype=2, sex=sex.value, **case_ps
        ),
        2: model["init_parameter_values"](
            amp=0.0, celltype=1, sex=sex.value, **case_ps
        ),
    }
    fun = {
        0: numba.njit(model["forward_generalized_rush_larsen"]),
        1: numba.njit(model["forward_generalized_rush_larsen"]),
        2: numba.njit(model["forward_generalized_rush_larsen"]),
    }
    v_index = {
        0: model["state_index"]("v"),
        1: model["state_index"]("v"),
        2: model["state_index"]("v"),
    }

    # Surface to volume ratio
    chi = 200.0 * ureg("mm**-1")
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

    s_l = (0.24 * ureg("S/m") / chi).to("uA/mV").magnitude
    s_t = (0.0456 * ureg("S/m") / chi).to("uA/mV").magnitude

    # M = beat.conductivities.define_conductivity_tensor(chi=chi, f0=data.fiber)
    f0 = data.fiber
    M = s_l * ufl.outer(f0, f0) + s_t * (ufl.Identity(3) - ufl.outer(f0, f0))

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

    # Simulate 5 beats
    T = 5000

    t = 0.0
    dt = 0.05
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

    vname = outdir / "voltage.xdmf"
    if vname.is_file() and comm.rank == 0:
        vname.unlink()
        vname.with_suffix(".h5").unlink()

    state_names = {}
    for marker in ode._values.keys():
        state_name = outdir / f"state_{marker}_{comm.rank}_{comm.size}.npy"
        state_names[marker] = state_name
        if state_name.is_file():
            state_name.unlink()

    leads = get_lead_positions()

    ecg_file = outdir / "extracellular_potential.json"

    i = 0
    while t < T + 1e-12:
        if i % 20 == 0:
            # Every ms
            v = solver.pde.state.vector().get_local()
            print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() = }")

            save_ecg(comm, t, solver.pde.state, ecg_file, leads)

        if i % 100 == 0:
            # Every 5 ms
            with dolfin.XDMFFile(comm, vname.as_posix()) as xdmf:
                xdmf.write_checkpoint(
                    solver.pde.state,
                    "V",
                    float(t),
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )
            for marker, value in ode._values.items():
                np.save(state_names[marker], value)

        solver.step((t, t + dt))
        i += 1
        t += dt


if __name__ == "__main__":
    # main(sex=Sex.male, case=Case.control)
    # main(sex=Sex.female, case=Case.control)
    main(sex=Sex.female, case=Case.dofe)
    main(sex=Sex.male, case=Case.dofe)
