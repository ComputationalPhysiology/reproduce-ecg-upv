from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_ecg(outdir: Path, ax1):
    if not (outdir / "ecg.csv").exists():
        print(f"Skipping {outdir.name}: no ecg.csv file")
        return None, None
    print(f"Plotting {outdir.name}")
    df = pd.read_csv(outdir / "ecg.csv")

    fig, ax2 = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
    for ax in [ax2, ax1]:
        l, = ax[0, 0].plot(df["time"], df["LA"])
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

        for axi in ax.flatten():
            axi.grid()

    fig.tight_layout()
    fig.savefig(outdir / "ecg.png")
    plt.close(fig)
    return l, df

def main():

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    all_ecg_file = Path("results-profile2") / "all_ecg.xlsx"
    lines = []
    labels = []
    with pd.ExcelWriter(all_ecg_file) as writer:
        for dir in Path("results-profile2").iterdir():

            l, df = plot_ecg(dir, ax)
            if df is None:
                continue

            df.to_excel(writer, sheet_name=dir.name[:29], index=False)
            lines.append(l)
            labels.append(dir.name)

    fig.legend(lines, labels, loc="center right")
    fig.subplots_adjust(right=0.85)
    fig.savefig("results-profile2/all_ecg.png", dpi=500)
    plt.close(fig)

if __name__ == "__main__":
    main()