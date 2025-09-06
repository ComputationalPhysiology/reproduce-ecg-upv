import time
from textwrap import dedent
from pathlib import Path
import subprocess as sp
from utils import Case

template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{sex}-{case}"
#SBATCH --partition={partition}
#SBATCH --time=10-00:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --output=slurm-output/%j-%x-stdout.txt
#SBATCH --error=slurm-output/%j-%x-stderr.txt

conda activate fenicsx-v09

ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results-profile{profile}/{sex}-{case}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

CONDA_PREFIX=/home/henriknf/miniforge3/envs/fenicsx-v09
$CONDA_PREFIX/bin/mpirun -n {ntasks} $CONDA_PREFIX/bin/python ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case} --profile {profile}
# Move log file to results folder
mv slurm-output/${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    i = 0
    profile = 1
    for sex in ["male", "female"]:
        for case in [c.name for c in Case]:

            outdir = Path(f"results-profile{profile}") / f"{sex}-{case}"
            print(outdir)
            if (outdir / "ode_state.h5").exists():
                print("Skipping")
                continue

            job_file = Path("tmp_job.sbatch")
            job_file.write_text(
                template.format(
                    sex=sex, 
                    case=case,
                    ntasks=32,
                    partition="defq",
                    profile=profile,
                )
            )
            sp.run(["sbatch", job_file.as_posix()])
            i += 1
            time.sleep(3)
            job_file.unlink()
      

if __name__ == "__main__":
    main()