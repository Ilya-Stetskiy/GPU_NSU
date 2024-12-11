#!/bin/bash
#SBATCH -J cuda_task # Job name
#for big gpuserv: #SBATCH -p compclass
#SBATCH -p gpuserv
#SBATCH -e cuda_tasks.%j.err
#SBATCH -o cuda_tasks.%j.out # Name of stdout output file (%j expands to %jobId)
#SBATCH -N 1 # Total number of nodes requested
#SBATCH -t 24:00:00 # Run time (hh:mm:ss)

echo "node=`hostname`"
echo $SLURM_JOB_NODELIST

