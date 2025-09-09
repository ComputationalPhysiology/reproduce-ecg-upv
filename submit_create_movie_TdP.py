import time
from textwrap import dedent
from pathlib import Path
import subprocess as sp
from utils import Case
import pandas as pd

template = dedent(
    """#!/bin/bash
#SBATCH --job-name="movie-{sex}-{case}"
#SBATCH --partition={partition}
#SBATCH --time=10-00:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --output=slurm-output/%j-%x-stdout.txt
#SBATCH --error=slurm-output/%j-%x-stderr.txt

ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results-profile{profile}/{sex}-{case}-CTRL-initial-states
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

/home/henriknf/miniforge3/envs/fenicsx-v09/bin/python3 ${{ROOT}}/main_fenicsx.py viz -d ${{ROOT}}/hex-mesh -r ${{SCRATCH_DIRECTORY}}
# Move log file to results folder
mv slurm-output/${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    profile = 2
    for sex in ["male", "female"]:
        for case in ["Quinidine_TdP", "Clozapine_TdP"]:

            outdir = Path(f"results-profile{profile}") / f"{sex}-{case}-CTRL-initial-states"
            if (outdir / "voltage.mp4").exists():
                print(f"Skipping {outdir.name}: already exists")
                continue
            if not (outdir / "ecg.csv").exists():
                continue
            df = pd.read_csv(outdir / "ecg.csv")
            if len(df["time"]) < 5000:
                print(f"Skipping {outdir.name}: not finished")
                continue

            print(f"Creating movie for {outdir.name}")
            job_file = Path("tmp_job.sbatch")
            job_file.write_text(
                template.format(
                    sex=sex, 
                    case=case,
                    ntasks=1,
                    partition="defq",
                    profile=profile
                )
            )
            sp.run(["sbatch", job_file.as_posix()])
            time.sleep(3)
            # exit()
            job_file.unlink()
            # exit()

# def run_all():
#     for sex in ["male", "female"]:
#         for case in [c.name for c in Case]:
#             outdir = Path("results") / f"{sex}-{case}"
#             outdir.mkdir(parents=True, exist_ok=True)
#             import main_fenicsx
#             main_fenicsx.main([
#                 "run", "-d", 
#                 "hex-mesh", "-o", str(outdir),"--sex", sex,"--case",case, "-r"
#             ])

if __name__ == "__main__":
    main()    
    # run_all()