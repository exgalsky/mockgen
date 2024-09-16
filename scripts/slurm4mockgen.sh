#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --job-name=xgsm-kappa
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gres=gpu:4

export SLURM_CPU_BIND="cores"

module unload cudatoolkit
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12
module load cray-mpich craype-accel-nvidia80

module use /pscratch/sd/s/shamikg/xgsmenv/cuda12-0.0.1/modulefiles
module load xgsmenv
source /pscratch/sd/s/shamikg/xgsmenv/cuda12-0.0.1/conda/bin/activate

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XGSMENV_NGPUS=4
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

srun -n 4 mockgen mockgen4nyx_test --N 64 --Niter 1 --laststep writeics --icw --zInit 50