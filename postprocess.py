from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json


def plot_ecg():
    fig, ax = plt.subplots(3, 1, sharex=True)

    outdir = Path("results-simula")
    outdir.mkdir(exist_ok=True, parents=True)

    for sex in ["male", "female"]:
        for tag in ["control", "dofe"]:
            resultsdir = Path(f"results-{sex}-{tag}")
            ecg_file = resultsdir / "extracellular_potential.json"
            if not ecg_file.is_file():
                continue
            data = json.loads(ecg_file.read_text())
            df = pd.DataFrame(data)
            df.to_csv(outdir / f"{sex}_{tag}.csv")

            ax[0].plot(df["time"].to_numpy(), df["I"].to_numpy(), label=sex)
            ax[1].plot(df["time"].to_numpy(), df["II"].to_numpy(), label=sex)
            ax[2].plot(df["time"].to_numpy(), df["III"].to_numpy(), label=sex)
        ax[0].set_title("I")
        ax[1].set_title("II")
        ax[2].set_title("III")
        ax[0].legend()
        fig.savefig(outdir / "ecg.png")


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def compare_simula_upv():
    simula = Path("results-simula")
    upv = Path("results-upv")
    outdir = Path("results-compare")
    outdir.mkdir(exist_ok=True, parents=True)

    tagmap = {"control": "ctrl", "dofe": "dofe"}

    for sex in ["male", "female"]:
        for tag in ["control", "dofe"]:
            for case in ["normalized", "original"]:
                f = normalize if case == "normalized" else lambda x: x
                fig, ax = plt.subplots(3, 1, sharex=True)

                simula_file = simula / f"{sex}_{tag}.csv"
                if simula_file.is_file():
                    df_simula = pd.read_csv(simula_file)
                    time = df_simula["time"][-1000:] - df_simula["time"].iloc[-1000]
                    ax[0].plot(
                        time,
                        f(df_simula["I"][-1000:]),
                        label="simula",
                    )
                    ax[1].plot(
                        time,
                        f(df_simula["II"][-1000:]),
                        label="simula",
                    )
                    ax[2].plot(
                        time,
                        f(df_simula["III"][-1000:]),
                        label="simula",
                    )

                upv_file = upv / f"{sex}_{tagmap[tag]}.csv"

                df_upv = pd.read_csv(
                    upv_file, names=["time", "I", "II", "III"], header=None
                )

                ax[0].plot(df_upv["time"], f(df_upv["I"]), label="upv")
                ax[1].plot(df_upv["time"], f(df_upv["II"]), label="upv")
                ax[2].plot(df_upv["time"], f(df_upv["III"]), label="upv")
                ax[0].set_title("I")
                ax[1].set_title("II")
                ax[2].set_title("III")
                ax[0].legend()
                fig.savefig(outdir / f"{sex}_{tag}_{case}.png")
                plt.close(fig)


if __name__ == "__main__":
    plot_ecg()
    compare_simula_upv()
