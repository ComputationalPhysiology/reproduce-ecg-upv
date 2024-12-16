from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json


def plot_ecg():
    fig, ax = plt.subplots(3, 1, sharex=True)

    outdir = Path("results-simula")
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in ["control", "dofe"]:
        for sex in ["male", "female"]:
            resultsdir = Path(f"results-{sex}-{tag}")
            ecg_file = resultsdir / "extracellular_potential.json"
            if not ecg_file.is_file():
                continue
            data = json.loads(ecg_file.read_text())
            df = pd.DataFrame(data)
            df.to_csv(outdir / f"{sex}_{tag}.csv", index=False)

            ax[0].plot(
                df["time"].to_numpy(), df["I"].to_numpy(), label=f"{sex} ({tag})"
            )
            ax[1].plot(
                df["time"].to_numpy(), df["II"].to_numpy(), label=f"{sex} ({tag})"
            )
            ax[2].plot(
                df["time"].to_numpy(), df["III"].to_numpy(), label=f"{sex} ({tag})"
            )
        ax[0].set_title("I")
        ax[1].set_title("II")
        ax[2].set_title("III")
        ax[0].legend()
        fig.savefig(outdir / "ecg-new.png")


if __name__ == "__main__":
    plot_ecg()
