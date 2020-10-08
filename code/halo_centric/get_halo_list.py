import numpy as np
import pandas as pd
import sys
import os
import copy
import argparse
import pdb

parser = argparse.ArgumentParser(description='Find most massive in a given tree.',
 usage= 'python')

# parser.add_argument('--treefile', type=str)
parser.add_argument('--halosdir', type=str, help='Directory path for halos saved data')
parser.add_argument('--simname', type=str, help='Simulation name')
parser.add_argument('--rundir', type=str, help='Directory name containing the snapshot binaries')

args = parser.parse_args()

halosdir = '/scratch/aseem/halos' if args.halosdir is None else args.halosdir
rundir = 'r1' if args.rundir is None else args.rundir
simname = 'bdm_cdm1024' if args.simname is None else args.simname

treesdir = os.path.join(halosdir, simname, rundir)

outdir = os.path.join('/scratch/cprem/sims',simname,rundir,'halo_centric','halos_list')
os.makedirs(outdir, exist_ok=True)
print(outdir)

i = 200
treefile = os.path.join(treesdir, 'out_{0:03d}.trees'.format(i))
print(treefile)

halos = pd.read_csv(treefile, sep=r'\s+', header=0, skiprows=list(range(1,58)), index_col='Depth_first_ID(28)', engine='c')#usecols = [0,1,2])

halos = halos[halos['pid(5)']==-1]

# print(halos.head(), '\n\n', halos.columns, '\n\n', halos.info(),'\n\n', halos.memory_usage())

logbin = 10**np.arange(10,14,0.3)
grouplist = list(halos.groupby(pd.cut(halos['mvir(10)'], bins=logbin)) )

halos_select = grouplist[-1][1]
# pdb.set_trace()

# halos_select_all

filepath = os.path.join(outdir,'halos_select')#_{0:03d}'.format(i))
halos_select.to_csv(filepath, sep='\t')

while i>0:
    i -= 1
    treefile = os.path.join(treesdir, 'out_{0:d}.trees'.format(i))
    print(treefile)
    halos = pd.read_csv(treefile, sep=r'\s+', header=0, skiprows=list(range(1,58)), index_col='Depth_first_ID(28)', engine='c')#usecols = [0,1,2])
    halos = halos[halos['pid(5)']==-1]

    # halos_select_previous = copy.deepcopy(halos_select)
    halos_to_look = halos_select.index + 1
    # print(list(halos.columns))

    halos_to_select = []
    for Depth_ID in halos_to_look:
        if Depth_ID in halos.index:
            halos_to_select.append(Depth_ID)

    halos_select = halos.loc[halos_to_select]

    # pdb.set_trace()

    # for Depth_ID in halos_to_look:
    #     try:
    #         halos_select.append(halos.loc[Depth_ID])
    #         print(halos.loc[Depth_ID])

    #     except KeyError:
    #         print('keyerror')
    #         pass

    # filepath = os.path.join(outdir,'halos_select_{0:03d}'.format(i))
    halos_select.to_csv(filepath, sep='\t', mode='a', header=False)
    






