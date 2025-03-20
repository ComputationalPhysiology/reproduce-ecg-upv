import time
from textwrap import dedent
from pathlib import Path
import subprocess as sp

template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{sex}-{case}"
#SBATCH --partition={partition}
#SBATCH --time=6-00:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --output=slurm-output/%j-%x-stdout.txt
#SBATCH --error=slurm-output/%j-%x-stderr.txt

ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results/{sex}-{case}/${{SLURM_JOBID}}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

PYTHON=/home/henriknf/miniforge3/envs/fenicsx-upv/bin/python

srun -n 8 ${{PYTHON}} ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case}
# Move log file to results folder
mv ${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    job_file = Path("tmp_job.sbatch")
    job_file.write_text(
        template.format(
            sex="female",
            case="control",
            ntasks=8,
            partition="slowq"
        )
    )
    sp.run(["sbatch", job_file.as_posix()])
    job_file.unlink()

if __name__ == "__main__":
    main()