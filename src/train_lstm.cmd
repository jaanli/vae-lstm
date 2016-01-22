#!/bin/bash
# GPU job for 1 week
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -t 168:00:00
# sends mail when process begins, and
# when it ends. Make sure you define your email
# address.
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jaan.altosaar@gmail.com

module load openmpi
module load python
module load cudatoolkit/7.0 cudann
sh ./ti_run.sh
