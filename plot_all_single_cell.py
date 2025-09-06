from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ORdmm

def plot_single_cell(outdir: Path, figdir: Path):

    dt = 1.0
    BCL = 1000
    t = np.arange(0, BCL, dt)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    all_states = []
    vs = [t]
    cas = [t]
    for i, cell_type in enumerate(["endo", "mid", "epi"]):
        celldir = outdir / "steady-states-0D" / cell_type
        files = list(celldir.glob("*.npy"))

        if len(files) == 0:
            print(f"No files found in {celldir}. Skipping.")
            return None, None, None

        states_file = next(iter(filter(lambda x: x.stem.startswith("steady_states"), files)))
        tracked_file = next(iter(filter(lambda x: x.stem.startswith("tracked_values"), files)))

        # states = np.load(states_file, allow_pickle=True).item()
        # tracked = np.load(tracked_file, allow_pickle=True).item()
        states = np.load(states_file)

        tracked = np.load(tracked_file)
        all_states.append(dict(zip(ORdmm.state.keys(), states)))

        v = tracked[-len(t):, 0]
        ca = tracked[-len(t):, 1]
        vs.append(v)
        cas.append(ca)


        ax[0].plot(t, v, label=cell_type)
        ax[1].plot(t, ca, label=cell_type)

    ax[0].set_title(outdir.name)
    ax[0].set_ylabel("Voltage (mV)")
    ax[1].set_ylabel("Calcium (mM)")
    ax[1].set_xlabel("Time (ms)")
    for axi in ax:
        axi.legend()
        axi.grid()
    fig.savefig(figdir / f"{outdir.name}.png")
    plt.close(fig)

    df = pd.DataFrame(all_states, index=["endo", "mid", "epi"])

    df_v = pd.DataFrame(np.array(vs).T, columns=["time", "endo", "mid", "epi"])
    df_ca = pd.DataFrame(np.array(cas).T, columns=["time", "endo", "mid", "epi"])
    return df, df_v, df_ca

def main():

    outdir = Path("results_profile1") / "single_cell"
    outdir.mkdir(parents=True, exist_ok=True) 
    all_v = outdir / "all_voltage.xlsx"
    all_ca = outdir / "all_calcium.xlsx"
    all_states = outdir / "all_initial_states.xlsx"
    lines = []
    labels = []
    with pd.ExcelWriter(all_v) as writer_voltage, \
         pd.ExcelWriter(all_ca) as writer_calcium, \
         pd.ExcelWriter(all_states) as writer_states:

        for dir in Path("results").iterdir():
            print(dir)
            df, df_v, df_ca = plot_single_cell(dir, outdir)
            if df is None:
                continue
            # Save to Excel files
            df.to_excel(writer_states, sheet_name=dir.name, index=True)
            df_v.to_excel(writer_voltage, sheet_name=dir.name, index=False)
            df_ca.to_excel(writer_calcium, sheet_name=dir.name, index=False)

      

if __name__ == "__main__":
    main()