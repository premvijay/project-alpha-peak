#!/bin/bash
# cd $PBS_O_WORKDIR
python ./assign_density.py --simdir /scratch/aseem/sims/ --simname scm1024 --rundir r1 --snap_i $1 --scheme TSC \
     --grid_size 512 --Pk --interlace --slice2D --outdir /scratch/cprem/sims/ >> out_file1.txt
