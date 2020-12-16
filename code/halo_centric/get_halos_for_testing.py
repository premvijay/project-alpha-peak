import numpy as np
import pandas as pd
import sys
import os
import copy
import argparse
import pdb
from time import time

parser = argparse.ArgumentParser(description='Find most massive in a given tree.',
 usage= 'python')

# parser.add_argument('--treefile', type=str)
parser.add_argument('--halosdir', type=str, help='Directory path for halos saved data')
parser.add_argument('--simname', type=str, help='Simulation name')
parser.add_argument('--rundir', type=str, help='Directory name containing the snapshot binaries')

# parser.add_argument('--select', type=str, default='normal')
parser.add_argument('--M_around', type=float, default=3e12)
parser.add_argument('--max_halos', type=int, default=300)

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

# logbin = 10**np.arange(10,14,0.3)
# grouplist = list(halos.groupby(pd.cut(halos['mvir(10)'], bins=logbin)) )

# halos_select = grouplist[-1][1]

# if args.select == 'dwarf':
#     M_bin = 2e11
# elif args.select == 'normal':


# for M_around in args.M_around_list:
t_now = time()

# def logbin(val, cen=args.M_around):
#     return np.fabs(np.log10(val) - np.log10(cen))

# halos_select = halos.sort_values('mvir(10)', key=logbin).iloc[:args.max_halos]

halos['diff_bin'] = np.fabs(np.log10(halos['mvir(10)']) - np.log10(args.M_around))

halos_select = halos.sort_values('diff_bin').iloc[:args.max_halos]

halos_select.drop('diff_bin', axis=1, inplace=True) 

t_bef, t_now = t_now, time()
print('total time for selecting root halos', t_now-t_bef)

halos_select['Snap_num(31)'] = int(i)

# pdb.set_trace()

# halos_select_all

filepath = os.path.join(outdir,'halos_select_{0:.1e}_{1:d}.csv'.format(args.M_around,args.max_halos))
halos_select.to_csv(filepath)

while i>1:
    i -= 1
    treefile = os.path.join(treesdir, 'out_{0:d}.trees'.format(i))
    if not os.path.exists(treefile):
        print('leaf reached')
        break
    else:
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

    halos_select['Snap_num(31)'] = int(i)
    halos_select.to_csv(filepath, mode='a', header=False)

    # pdb.set_trace()

    # for Depth_ID in halos_to_look:
    #     try:
    #         halos_select.append(halos.loc[Depth_ID])
    #         print(halos.loc[Depth_ID])

    #     except KeyError:
    #         print('keyerror')
    #         pass

    # filepath = os.path.join(outdir,'halos_select_{0:03d}'.format(i))
    

