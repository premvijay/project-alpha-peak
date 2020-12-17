import numpy as np
import pandas as pd
import sys
import os
import copy
import argparse
import pdb
from time import time

from halo_tools.trace_halo_history import mmp_branch
from gadget_tools import Snapshot

parser = argparse.ArgumentParser(description='Find most massive in a given tree.',
 usage= 'python')

# parser.add_argument('--treefile', type=str)
parser.add_argument('--halosdir', type=str, default='/scratch/aseem/halos', help='Directory path for halos saved data')
parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Simulation name')
parser.add_argument('--rundir', type=str, default='r1', help='Directory name containing the snapshot binaries')
parser.add_argument('--simdir', type=str, default='/scratch/aseem/sims', help='Directory path for halos saved data')
# parser.add_argument('--select', type=str, default='normal')
parser.add_argument('--M_range', type=float, nargs=2, default=[2.9e12,3.1e12])
parser.add_argument('--Gam_range', type=float, nargs=2, default=[0.5,1])
parser.add_argument('--max_halos', type=int, default=300)

args = parser.parse_args()

halosdir = args.halosdir
rundir = args.rundir
simname = args.simname

treesdir = os.path.join(halosdir, simname, rundir)

outdir = os.path.join('/scratch/cprem/sims',simname,rundir,'halo_centric','halos_list')
os.makedirs(outdir, exist_ok=True)
print(outdir)

snapdir = os.path.join(args.simdir, args.simname, args.rundir)


i = 200


treefile = os.path.join(treesdir, 'out_{0:03d}.trees'.format(i))
print(treefile)
snapfile = os.path.join(snapdir, f'snapshot_{i:03d}.0')
snap = Snapshot(snapfile=snapfile)

halos = pd.read_csv(treefile, sep=r'\s+', header=0, skiprows=list(range(1,58)), engine='c')#usecols = [0,1,2])
halos = halos[halos['pid(5)']==-1]
# halos.set_index('Depth_first_ID(28)', inplace=True)

t_now = time()

accr_string = '1*Tdyn(64)'


# selection criteria
# mass_range = (2.9e12, 3.1e12)
# Gamma_range = (0.5, 1)

# halos_sp_accr_rate = 

halos_select_mass = halos[halos['mvir(10)'].between(*args.M_range)]

halos_select_mass['Gamma'] = halos_select_mass[f'Acc_Rate_{accr_string:s}']/halos_select_mass['mvir(10)'] * 9.778e9 / snap.Hubble_param

halos_select_gamma = halos_select_mass[halos_select_mass['Gamma'].between(*args.Gam_range)]

# halos['diff_bin'] = np.fabs(np.log10(halos['mvir(10)']) - np.log10(args.M_around))

# halos_select = halos.sort_values('diff_bin').iloc[:args.max_halos]

# halos_select.drop('diff_bin', axis=1, inplace=True) 
# print(list(halos_select_gamma.index), '\n', np.random.choice(list(halos_select_gamma.index), size=args.max_halos, replace=False).tolist())
num_halos = min(halos_select_gamma.shape[0], args.max_halos)
print(f'number of halos is {num_halos}')
# halos_select = halos_select_gamma.iloc[np.random.choice(list(halos_select_gamma.index), size=num_halos, replace=False).tolist()]
halos_select = halos_select_gamma.sample(n=num_halos)


t_bef, t_now = t_now, time()
print('total time for selecting root halos', t_now-t_bef)

halos_select['Snap_num(31)'] = int(i)

filepath = os.path.join(outdir,'halos_select_M_{0:.2g}to{1:.2g}_G_{2:.2g}to{3:.2g}.csv'.format(*args.M_range, *args.Gam_range))

halos_select.set_index('Depth_first_ID(28)', inplace=True)
halos_select.to_csv(filepath)

mmp_branch(filepath, treesdir, 1)

    

