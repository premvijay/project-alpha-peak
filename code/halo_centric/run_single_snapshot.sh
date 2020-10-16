#!/bin/bash
# cd $PBS_O_WORKDIR
python ./assign_density.py --simdir /scratch/aseem/sims/ --simname bdm_cdm1024 --rundir r1 --snap_i $1 \
     --M_around $2 --max_halos 500 --downsample 8 --scheme TSC \
     --grid_size 512 --align --slice2D --outdir /scratch/cprem/sims/ | tee -a out_file1.txt
