from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json


def plot_ecg():
    fig, ax = plt.subplots(3, 1, sharex=True)

    for sex in ["male", "female"]:
        outdir = Path(f"results-{sex}")
        data = json.loads((outdir / "extracellular_potential.json").read_text())
        df = pd.DataFrame(data)

        ax[0].plot(df["time"].to_numpy(), df["I"].to_numpy(), label=sex)
        ax[1].plot(df["time"].to_numpy(), df["II"].to_numpy(), label=sex)
        ax[2].plot(df["time"].to_numpy(), df["III"].to_numpy(), label=sex)
    ax[0].set_title("I")
    ax[1].set_title("II")
    ax[2].set_title("III")
    ax[0].legend()
    fig.savefig("ecg.png")


if __name__ == "__main__":
    plot_ecg()
