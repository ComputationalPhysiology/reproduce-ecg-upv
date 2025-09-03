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

#module use /cm/shared/spack-modules/modulefiles
#module load spack/0.23.1
#umask 0002
#spack env activate fenicsx-stable-milanq-openmpi
#module load openmpi/gcc/64/4.1.5
#export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH
#export MCA_btl_openib_allow_ib=1
# conda activate fenicsx-v09

ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results/{sex}-{case}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"


/home/henriknf/miniforge3/envs/fenicsx-v09/bin/python3 ${{ROOT}}/main_fenicsx.py viz -d ${{ROOT}}/hex-mesh -r ${{SCRATCH_DIRECTORY}}
# Move log file to results folder
mv slurm-output/${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    for sex in ["male", "female"]:
        for case in [c.name for c in Case]:

            outdir = Path("results") / f"{sex}-{case}"
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
                    partition="defq"
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