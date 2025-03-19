from pathlib import Path
from typing import NamedTuple, Sequence, TYPE_CHECKING
import logging
import argparse
import shutil


from utils import Sex, Case, case_parameters, get_lead_positions

if TYPE_CHECKING:
    import dolfinx

here = Path(__file__).parent.absolute()

logger = logging.getLogger(__name__)


def convert_data(
    input: Path = Path("ModVent_PAP_hexaVol_0-4mm_17endo-42epi_Labeled_full (1).vtk"),
    output: Path = Path("hex-mesh"),
):
    import pyvista as pv
    from mpi4py import MPI
    import dolfinx
    import adios4dolfinx

    logger.info(f"Reading mesh file {input}")
    reader = pv.get_reader(input)
    vtk_mesh = reader.read()

    output.mkdir(exist_ok=True)
    data_file = output / "mesh.xdmf"
    logger.info(f"Save mesh to XDMF file {data_file}")
    pv.save_meshio(output / "mesh.xdmf", vtk_mesh)

    comm = MPI.COMM_WORLD
    logger.info("Load mesh into dolfinx")
    with dolfinx.io.XDMFFile(comm, data_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    logger.info("Create endo_epi function")
    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    endo_epi = dolfinx.fem.Function(V, name="EndoToEpi")

    endo_epi.x.array[:] = vtk_mesh.point_data.get_array("EndoToEpi")[
        mesh.geometry.input_global_indices
    ]
    logger.info("Save endo_epi function for visualization")
    with dolfinx.io.VTXWriter(
        mesh.comm, output / "endo_epi_viz.bp", [endo_epi], engine="BP5"
    ) as vtx:
        vtx.write(0.0)
    logger.info("Save endo_epi function for simulation")
    adios4dolfinx.write_function_on_input_mesh(
        output / "endo_epi.bp", endo_epi, time=0.0, name="endo_epi"
    )

    logger.info("Create fiber function")
    W = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    fiber = dolfinx.fem.Function(W, name="Fiber")
    fiber_data = vtk_mesh.cell_data.get_array("FibreOrientation")

    inds = mesh.topology.original_cell_index
    fiber.x.array[::3] = fiber_data[inds, 0]
    fiber.x.array[1::3] = fiber_data[inds, 1]
    fiber.x.array[2::3] = fiber_data[inds, 2]
    logger.info("Save fiber function for visualization")
    with dolfinx.io.VTXWriter(
        mesh.comm, output / "fiber_viz.bp", [fiber], engine="BP5"
    ) as vtx:
        vtx.write(0.0)

    logger.info("Save fiber function for simulation")
    adios4dolfinx.write_function_on_input_mesh(
        output / "fiber.bp", fiber, time=0.0, name="fiber"
    )

    logger.info("Create stimulus")
    create_stimulus(mesh, output)


class Geometry(NamedTuple):
    mesh: "dolfinx.mesh.Mesh"
    fiber: "dolfinx.fem.Function"
    endo_epi: "dolfinx.fem.Function"
    I_stim: "dolfinx.fem.Function"

    def update_stimulus(self, datadir, t, stim_duration=2, amplitude=10.0):
        import adios4dolfinx

        logger.debug(f"Update stimulus at time {t}")
        T = int(t)
        self.I_stim.x.array[:] = 0.0
        # Last time step is 48
        if T >= 49:
            return
        T_start = max(T - stim_duration, 0)
        T_end = min(T + stim_duration, 49)
        Istim_tmp = dolfinx.fem.Function(self.I_stim.function_space)
        for t in range(T_start, T_end):
            adios4dolfinx.read_function(
                datadir / "Istim.bp", Istim_tmp, name="Istim", time=t
            )
            self.I_stim.x.array[:] += Istim_tmp.x.array[:]

        # Set the values to be the amplitude
        self.I_stim.x.array[self.I_stim.x.array > 0] = amplitude


def load_data(datadir) -> Geometry:
    from mpi4py import MPI
    import dolfinx
    import adios4dolfinx

    comm = MPI.COMM_WORLD
    logger.info("Load mesh")
    with dolfinx.io.XDMFFile(comm, datadir / "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    endo_epi = dolfinx.fem.Function(V, name="EndoToEpi")

    logger.info("Load endo_epi function")
    adios4dolfinx.read_function(datadir / "endo_epi.bp", endo_epi, name="endo_epi")

    W = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    fiber = dolfinx.fem.Function(W, name="Fiber")
    logger.info("Load fiber function")
    adios4dolfinx.read_function(datadir / "fiber.bp", fiber, name="fiber")

    W = dolfinx.fem.functionspace(mesh, ("DG", 0))

    Istim = dolfinx.fem.Function(W, name="Istim")

    return Geometry(mesh, fiber, endo_epi, I_stim=Istim)


def create_stimulus(mesh, datadir):
    import pandas as pd
    import dolfinx
    import numpy as np
    import adios4dolfinx

    logger.info("Create stimulus")
    df = pd.read_csv("tact_pmj.csv")

    W = dolfinx.fem.functionspace(mesh, ("DG", 0))

    Istim = dolfinx.fem.Function(W, name="Istim")

    shutil.rmtree(datadir / "stim_viz.bp", ignore_errors=True)
    shutil.rmtree(datadir / "Istim.bp", ignore_errors=True)
    vtx = dolfinx.io.VTXWriter(
        mesh.comm, datadir / "stim_viz.bp", [Istim], engine="BP5"
    )

    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim, padding=0.01)
    for t in df.Activation.unique():
        logger.info(f"\nCreate stimulus at time {t}")
        points_t = df[df["Activation"] == t]
        inds = []
        for i, row in points_t.iterrows():
            points = np.array([row.Points_0, row.Points_1, row.Points_2])

            collisions = dolfinx.geometry.compute_collisions_points(tree, points)
            if len(collisions.array) == 0:
                continue
            cell = collisions.array[0]
            inds.append(cell)
        logger.info(f"Number of points {len(inds)}")

        Istim.x.array[np.array(inds)] = 1.0
        adios4dolfinx.write_function_on_input_mesh(
            datadir / "Istim.bp", Istim, time=t, name="Istim"
        )
        vtx.write(t)
        Istim.x.array[:] = 0.0


def load_stimulus(Istim, datadir, t, stim_duration=2):
    import adios4dolfinx
    import dolfinx

    T = int(t)
    Istim.x.array[:] = 0.0
    if T >= 50:
        return
    T_start = max(T - stim_duration, 0)
    T_end = min(T + stim_duration, 50)
    Istim_tmp = dolfinx.fem.Function(Istim.function_space)
    for t in range(T_start, T_end):
        adios4dolfinx.read_function(
            datadir / "Istim.bp", Istim_tmp, name="Istim", time=t
        )
        Istim.x.array[:] += Istim_tmp.x.array[:]


def run(
    outdir: Path,
    datadir: Path,
    sex: Sex = Sex.male,
    case: Case = Case.control,
):
    import dolfinx
    import adios4dolfinx

    import scifem
    import gotranx
    import numba
    import ufl

    import beat

    geo = load_data(datadir)
    outdir.mkdir(exist_ok=True)

    comm = geo.mesh.comm

    save_every_ms = 1.0
    dt = 0.05
    save_freq = round(save_every_ms / dt)

    case_ps = case_parameters(case)
    module_path = Path("ORdmm_Land.py")
    if not module_path.is_file():
        ode = gotranx.load_ode(here / "ORdmm_Land.ode")
        code = gotranx.cli.gotran2py.get_code(
            ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen]
        )
        if comm.rank == 0:
            module_path.write_text(code)

    comm.barrier()

    import ORdmm_Land

    model = ORdmm_Land.__dict__

    init_states = {
        0: beat.single_cell.get_steady_state(
            fun=model["generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](
                celltype=0, sex=sex.value, **case_ps
            ),
            outdir=outdir / "steady-states-0D" / "mid",
            BCL=1000,
            nbeats=500,
            track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
            dt=0.05,
        ),
        1: beat.single_cell.get_steady_state(
            fun=model["generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](
                celltype=2, sex=sex.value, **case_ps
            ),
            outdir=outdir / "steady-states-0D" / "endo",
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
            fun=model["generalized_rush_larsen"],
            init_states=model["init_state_values"](),
            parameters=model["init_parameter_values"](
                celltype=1, sex=sex.value, **case_ps
            ),
            outdir=outdir / "steady-states-0D" / "epi",
            BCL=1000,
            nbeats=500,
            track_indices=[model["state_index"]("v"), model["state_index"]("cai")],
            dt=0.05,
        ),
    }
    mesh_unit = "mm"

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
        0: numba.njit(model["generalized_rush_larsen"]),
        1: numba.njit(model["generalized_rush_larsen"]),
        2: numba.njit(model["generalized_rush_larsen"]),
    }
    v_index = {
        0: model["state_index"]("v"),
        1: model["state_index"]("v"),
        2: model["state_index"]("v"),
    }

    # Surface to volume ratio
    chi = 200.0 * beat.units.ureg("mm**-1")
    # Membrane capacitance
    C_m = 0.01 * beat.units.ureg("uF/mm**2")

    s_l = (0.24 * beat.units.ureg("S/m") / chi).to("uA/mV").magnitude
    s_t = (0.0456 * beat.units.ureg("S/m") / chi).to("uA/mV").magnitude

    f0 = geo.fiber
    M = s_l * ufl.outer(f0, f0) + s_t * (ufl.Identity(3) - ufl.outer(f0, f0))

    time = dolfinx.fem.Constant(geo.mesh, dolfinx.default_scalar_type(0.0))

    pde = beat.MonodomainModel(
        time=time,
        mesh=geo.mesh,
        M=M,
        I_s=geo.I_stim,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
    )

    V_ode = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    ode = beat.odesolver.DolfinMultiODESolver(
        v_ode=dolfinx.fem.Function(V_ode),
        v_pde=pde.state,
        markers=geo.endo_epi,
        num_states={i: len(s) for i, s in init_states.items()},
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        v_index=v_index,
    )

    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

    vtxfname = outdir / "v.bp"
    checkpointfname = outdir / "v_checkpoint.bp"

    # Make sure to remove the files if they already exist

    shutil.rmtree(vtxfname, ignore_errors=True)
    shutil.rmtree(checkpointfname, ignore_errors=True)
    vtx = dolfinx.io.VTXWriter(
        comm,
        vtxfname,
        [solver.pde.state],
        engine="BP4",
    )

    def save(t):
        logger.info(f"Save data at time {t:.3f}")
        vtx.write(t)
        adios4dolfinx.write_function_on_input_mesh(
            checkpointfname, solver.pde.state, time=t, name="v"
        )

    ode_state_file = outdir / "ode_state.xdmf"
    ode_state_file.unlink(missing_ok=True)
    ode_state_file.with_suffix(".h5").unlink(missing_ok=True)

    state_functions = ode.states_to_dolfin(names=model["state"].keys())
    xdmf = scifem.xdmf.XDMFFile(ode_state_file, state_functions)
    xdmf.write(0.0)

    num_beats = 5
    BCL = 1000.0
    for b in range(num_beats):
        t = 0.0
        i = 0
        while t < BCL + 1e-12:
            # Load stimulus every ms
            if i % int(1 / dt) == 0:
                geo.update_stimulus(datadir, t)

            if i % save_freq == 0:
                save(t)

            solver.step((t, t + dt))
            i += 1
            t += dt

        ode.assign_all_states(state_functions)
        xdmf.write(float(b + 1))


def get_outdir(sex: Sex = Sex.male, case: Case = Case.control, hexmesh: bool = True):
    if hexmesh:
        outdir = Path(f"hex-mesh-results-{sex.name}-{case.name}")
    else:
        outdir = Path(f"tet-mesh-results-{sex.name}-{case.name}")

    return outdir


def plot_single_ecg(
    sex: Sex = Sex.male, case: Case = Case.control, hexmesh: bool = True
):
    import pandas as pd
    import matplotlib.pyplot as plt

    outdir = get_outdir(sex=sex, case=case, hexmesh=hexmesh)

    df = pd.read_csv(outdir / "ecg.csv")
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True)

    ax[0, 0].plot(df["time"], df["LA"])
    ax[0, 0].set_title("LA")
    ax[0, 1].plot(df["time"], df["RA"])
    ax[0, 1].set_title("RA")
    ax[0, 2].plot(df["time"], df["LL"])
    ax[0, 2].set_title("LL")

    ax[1, 0].plot(df["time"], df["I"])
    ax[1, 0].set_title("I")
    ax[1, 1].plot(df["time"], df["II"])
    ax[1, 1].set_title("II")
    ax[1, 2].plot(df["time"], df["III"])
    ax[1, 2].set_title("III")

    fig.tight_layout()
    fig.savefig(outdir / "ecg.png")


def compare_ecg(plot_upv=True):
    import matplotlib.pyplot as plt
    import pandas as pd

    lines = []
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
    for hexmesh in [False, True]:
        outdir = get_outdir(hexmesh=hexmesh)
        df = pd.read_csv(
            outdir / "ecg.csv",
        )

        ax[0, 0].plot(df["time"], df["LA"] / 0.05)
        ax[0, 0].set_title("LA")
        ax[0, 1].plot(df["time"], df["RA"] / 0.05)
        ax[0, 1].set_title("RA")
        ax[0, 2].plot(df["time"], df["LL"] / 0.05)
        ax[0, 2].set_title("LL")

        ax[1, 0].plot(df["time"], df["I"] / 0.05)
        ax[1, 0].set_title("I")
        ax[1, 1].plot(df["time"], df["II"] / 0.05)
        ax[1, 1].set_title("II")
        (l,) = ax[1, 2].plot(df["time"], df["III"] / 0.05)
        ax[1, 2].set_title("III")
        lines.append(l)
    labels = ["Tet mesh", "Hex mesh"]

    if plot_upv:
        start_upv = 30
        upv = pd.read_csv("results-upv/male_ctrl.csv")

        time, I, II, III = upv.values.T

        ax[1, 0].plot(
            time[:500],
            0.17 * I[start_upv : start_upv + 500],
            linestyle="--",
            color="black",
        )
        ax[1, 1].plot(
            time[:500],
            0.17 * II[start_upv : start_upv + 500],
            linestyle="--",
            color="black",
        )
        (l,) = ax[1, 2].plot(
            time[:500],
            0.17 * III[start_upv : start_upv + 500],
            linestyle="--",
            color="black",
        )
        lines.append(l)
        labels.append("UPV*0.17 (shift 30 ms)")
        fname = "ecg-compare-upv.png"
        ncol = 3
    else:
        ncol = 2
        fname = "ecg-compare.png"

    lgd = fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.tight_layout()
    fig.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches="tight")


def compute_ecg(
    datadir: Path,
    outdir: Path,
):
    from mpi4py import MPI
    import pandas as pd
    import beat
    import ufl
    import adios4dolfinx
    import dolfinx

    geo = load_data(datadir)

    V_ode = dolfinx.fem.functionspace(geo.mesh, ("P", 1))
    v = dolfinx.fem.Function(V_ode)
    mesh_unit = "mm"
    C_m = 0.01 * beat.units.ureg("uF/mm**2")
    # Surface to volume ratio
    chi = 200.0 * beat.units.ureg("mm**-1")
    s_l = (0.24 * beat.units.ureg("S/m") / chi).to("uA/mV").magnitude
    s_t = (0.0456 * beat.units.ureg("S/m") / chi).to("uA/mV").magnitude

    f0 = geo.fiber
    M = s_l * ufl.outer(f0, f0) + s_t * (ufl.Identity(3) - ufl.outer(f0, f0))

    recv = beat.ecg.ECGRecovery(
        v=v,
        sigma_b=1.0,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        dx=None,
        M=M,
    )
    checkpointfname = outdir / "v_checkpoint.bp"
    time_stamps = adios4dolfinx.read_timestamps(
        comm=geo.mesh.comm, filename=checkpointfname, function_name="v"
    )
    leads = get_lead_positions()
    LA_form = recv.eval(point=leads["LA"])
    RA_form = recv.eval(point=leads["RA"])
    LL_form = recv.eval(point=leads["LL"])

    ecg_data = []
    for t in time_stamps:
        adios4dolfinx.read_function(checkpointfname, v, name="v", time=t)

        recv.solve()
        LA = geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(LA_form), op=MPI.SUM)
        RA = geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(RA_form), op=MPI.SUM)
        LL = geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(LL_form), op=MPI.SUM)

        I = LA - RA
        II = LL - RA
        III = LL - LA

        ecg_data.append(
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
    if geo.mesh.comm.rank == 0:
        df = pd.DataFrame(ecg_data)
        df.to_csv(outdir / "ecg.csv", index=False)


def get_parser():
    kwargs = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Reproduce UPV ECG", **kwargs)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Run parser
    run_parser = subparsers.add_parser(
        "run", help="Print information about the data", **kwargs
    )

    # See https://stackoverflow.com/a/46385352
    run_parser.add_argument(
        "-s",
        "--sex",
        default="male",
        type=str,
        choices=list(Sex.__members__),
        help="Sex",
    )
    run_parser.add_argument(
        "-c",
        "--case",
        default="control",
        type=str,
        choices=list(Case.__members__),
        help="Case",
    )
    run_parser.add_argument(
        "-d",
        "--datadir",
        type=Path,
        default=Path("hex-mesh"),
        help="Path to directory with mesh",
    )
    run_parser.add_argument("-o", "--outdir", type=Path, help="Output directory")

    # Convert parser
    convert_parser = subparsers.add_parser(
        "convert", help="Convert vtk data to dolfinx", **kwargs
    )
    convert_parser.add_argument("-i", "--input", type=Path, help="Input vtk file")
    convert_parser.add_argument("-o", "--output", type=Path, help="Output folder")

    # Compute ECG parser
    ecg_parser = subparsers.add_parser("ecg", help="Compute ECG", **kwargs)
    ecg_parser.add_argument(
        "-d", "--datadir", type=Path, help="Path to directory with mesh"
    )
    ecg_parser.add_argument("-o", "--outdir", type=Path, help="Output directory")

    return parser


def validate_enums(**kwargs):
    sex = kwargs.pop("sex")
    if sex is None:
        raise ValueError("Please specify sex")
    if sex not in Sex.__members__:
        raise ValueError(f"Invalid sex {sex}, please choose from {Sex.__members__}")
    sex_ = Sex[sex]

    case = kwargs.pop("case")
    if case is None:
        raise ValueError("Please specify case")
    if case not in Case.__members__:
        raise ValueError(f"Invalid case {case}, please choose from {Case.__members__}")
    case_ = Case[case]

    return sex_, case_, kwargs


def dispatch(parser, argv: Sequence[str] | None = None) -> int:
    args = vars(parser.parse_args(argv))
    logging.basicConfig(level=logging.DEBUG if args.pop("verbose") else logging.INFO)

    command = args.pop("command")
    if command is None:
        parser.error("Please specify a command")

    if command == "run":
        sex, case, args = validate_enums(**args)
        run(sex=sex, case=case, **args)
    if command == "convert":
        convert_data(**args)
    if command == "ecg":
        compute_ecg(**args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = get_parser()

    return dispatch(parser, argv)


if __name__ == "__main__":
    raise SystemExit(main())
