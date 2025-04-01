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
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

module use /cm/shared/spack-modules/modulefiles
module load spack/0.23.1
spack env activate fenicsx-stable-slowq
export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH


ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results/{sex}-{case}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

#srun -n {ntasks} python3 ${{ROOT}}/test_adios2.py
srun python3 ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case}
# Move log file to results folder
mv ${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    for sex in ["male", "female"]:
        for case in ["control", "dofe"]:
            
            job_file = Path("tmp_job.sbatch")
            job_file.write_text(
                template.format(
                    sex=sex, 
                    case=case,
                    ntasks=36,
                    partition="slowq"
                    # partition="xeongold16q"
                )
            )
            sp.run(["sbatch", job_file.as_posix()])
            job_file.unlink()
            # exit()
            time.sleep(10)

if __name__ == "__main__":
    main()