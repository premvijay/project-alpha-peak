#!/bin/bash
# cd $PBS_O_WORKDIR
pwd
module load anaconda3
export PYTHONPATH="$PYTHONPATH:/mnt/home/student/cprem/project-alpha-peak/code/library"
python ./assign_density.py --snapdir /scratch/aseem/sims/bdm_cdm1024/r1/ --snap_i $1 --scheme TSC \
     --grid_size 512 --Pk --slice2D --outdir /mnt/home/student/cprem/myscratch/bdm_cdm1024/