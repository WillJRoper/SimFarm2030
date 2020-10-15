
#!/bin/bash
# Tell SGE that we are using the bash shell
#$ -S /bin/bash

# Example file to create a task-farm of identical jobs on Apollo

# Do not limit the stacksize (the maximum memory a job can use)
ulimit -s unlimited
# Do not limit the number of open files a job can have
#ulimit -n unlimited
# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /lustre/scratch/astro/wjr21/SimFarm/SimFarm2030/core
# Created files will have rw permission for the group and r for the world
umask 002

# Set pathnames below relative to the current working directory
#$ -cwd
# Say which queue you want to submit to
#$ -q smp.q
# Join standard error to standard out
#$ -j y
# Give the job a name
#$ -N SimFarm-Daily
# Name and location of the output file
# SGE will only substitute certain variables here
#$ -o logs/$JOB_NAME.log
#$ -l h=!(node105|node104|node106)

module purge
module load easybuild/software
module load Anaconda3/2019.10

# Activate python environment
source activate simfarm-env

# Run jobs
echo "Starting job"
python model_daily.py 
echo "Finished job script"
