#!/bin/bash
# cd $PBS_O_WORKDIR
pwd
module load anaconda3
export PYTHONPATH="$PYTHONPATH:/mnt/home/student/cprem/project-alpha-peak/code/library"
python ./assign_density.py --snapdir /scratch/aseem/sims/bdm_cdm1024/r1/ --snap_i $1 --scheme CIC \
     --grid_size 512 --Pk --interlace --slice2D --outdir /scratch/cprem/sims/bdm_cdm1024/