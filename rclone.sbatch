#!/bin/bash
# Basic usage:
#
# $ sbatch hello.sbatch
#
# The job is queued and starts as soon as resources are available. The
# script is then executed on one of the allocated tasks, and standard
# output and standard error streams will be redirected to files that
# are prefixed by the job ID and job name. Commands prefixed with
# `srun' are executed on every task acquired by the job allocation.
#
# The sbatch options below allocate a single task on a single node,
# using a single CPU core with a one-hour time limit. To override
# these defaults, you can also supply sbatch options on the command
# line. For example:
#
# $ sbatch --cpus-per-task=32 --time=02:00:00 hello.sbatch
#SBATCH --job-name="rclone-reproduce-ecg-upv"
#SBATCH --partition=slowq
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-output/%j-%x-stdout.txt
#SBATCH --error=slurm-output/%j-%x-stderr.txt

# rclone copy --progress /global/D1/homes/henriknf/reproduce-ecg-upv/qt_intervals gdrive:reproduce-ecg-upv/qt_intervals/
rclone copy  --exclude "**/v_checkpoint.bp"  --progress /global/D1/homes/henriknf/reproduce-ecg-upv/results gdrive:reproduce-ecg-upv/