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

# module use /cm/shared/spack-modules/modulefiles
# module load spack/0.23.1
# umask 0002
# spack env activate fenicsx-stable-milanq-openmpi
# module load openmpi/gcc/64/4.1.5
# export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH
# export OMPI_MCA_btl_openib_allow_ib=1

conda activate fenicsx-v09

ROOT=/global/D1/homes/${{USER}}/reproduce-ecg-upv
SCRATCH_DIRECTORY=${{ROOT}}/results/{sex}-{case}
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

/home/henriknf/miniforge3/envs/fenicsx-v09/bin/mpirun -n {ntasks} /home/henriknf/miniforge3/envs/fenicsx-v09/bin/python ${{ROOT}}/main_fenicsx.py run -d ${{ROOT}}/hex-mesh -o ${{SCRATCH_DIRECTORY}} --sex {sex} --case {case}
# Move log file to results folder
mv slurm-output/${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)


def main():

    for sex in ["male", "female"]:
        for case in [c.name for c in Case]:
            # outdir = Path("results") / f"{sex}-{case}"
            # print(outdir)
            # import shutil
            # shutil.rmtree(outdir / "v_checkpoint.bp", ignore_errors=True)
            # (outdir / "ecg.csv").unlink(missing_ok=True)
            # (outdir / "log.txt").unlink(missing_ok=True)
            # (outdir / "ode_state.h5").unlink(missing_ok=True)
            # (outdir / "ode_state.xdmf").unlink(missing_ok=True)
            job_file = Path("tmp_job.sbatch")
            job_file.write_text(
                template.format(
                    sex=sex, 
                    case=case,
                    ntasks=64,
                    partition="milanq,genoaxq,defq"
                    # partition="xeongold16q"
                )
            )
            sp.run(["sbatch", job_file.as_posix()])
      
            time.sleep(3)
            job_file.unlink()

if __name__ == "__main__":
    main()