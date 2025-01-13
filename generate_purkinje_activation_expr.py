from pathlib import Path
import pandas as pd

df = pd.read_csv("tact_pmj.csv")

duration = 2.0
start = 0.0
stim_expr = []

for i, row in df.iterrows():
    stim_expr.append(
        f"(time > {row.Activation + start:.3f} && time < {start + row.Activation + duration:.3f} && "
        f"near(x[0], {row.Points_0:.3f}, tol) && "
        f"near(x[1], {row.Points_1:.3f}, tol) && "
        f"near(x[2], {row.Points_2:.3f}, tol))"
    )

expr = (
    f"time < {start + df.Activation.max() + duration} && ("
    + " || ".join(stim_expr)
    + ") ? stim_amp : 0.0"
)
Path("stimulation_expr.text").write_text(expr)
