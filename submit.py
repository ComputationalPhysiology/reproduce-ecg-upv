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
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

module use /cm/shared/spack-modules/modulefiles
module load spack/0.23.1
umask 0002
spack env activate fenicsx-stable-milanq-openmpi
module load openmpi/gcc/64/4.1.5
export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH


ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results/{sex}-{case}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

mpirun -n {ntasks} python3 ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case}
# Move log file to results folder
mv ${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    for sex in ["male", "female"]:
        for case in [c.name for c in Case]:
            job_file = Path("tmp_job.sbatch")
            job_file.write_text(
                template.format(
                    sex=sex, 
                    case=case,
                    ntasks=64,
                    partition="milanq"
                    # partition="xeongold16q"
                )
            )
            sp.run(["sbatch", job_file.as_posix()])
            job_file.unlink()
            # exit()
            time.sleep(10)

if __name__ == "__main__":
    main()