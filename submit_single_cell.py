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

ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results-profile{profile}/{sex}-{case}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

# mpirun -n {ntasks} python3 ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case} -r
/home/henriknf/miniforge3/envs/fenicsx-upv/bin/python3 ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case} -r --profile {profile}
# Move log file to results folder
mv slurm-output/${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    for sex in ["male", "female"]:
        # for case in [c.name for c in Case]:
        for case in ["Control", "Quinidine_TdP", "Clozapine_TdP"]:
            job_file = Path("tmp_job.sbatch")
            job_file.write_text(
                template.format(
                    sex=sex, 
                    case=case,
                    ntasks=1,
                    profile=2,
                    partition="defq"
                    # partition="xeongold16q"
                )
            )
            sp.run(["sbatch", job_file.as_posix()])
            job_file.unlink()
            # exit()
            time.sleep(3)

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