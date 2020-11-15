import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')


args = parser.parse_args()

halosdir = '/scratch/aseem/halos' if args.halosdir is None else args.halosdir

treesdir = os.path.join(halosdir, args.simname, args.rundir)

outdir = os.path.join('/scratch/cprem/sims',args.simname,args.rundir,'halo_centric','halos_list')
os.makedirs(outdir, exist_ok=True)
print(outdir)

i = 190
treefile = os.path.join(treesdir, 'out_{0:03d}.trees'.format(i))
print(treefile)

halos = pd.read_csv(treefile, sep=r'\s+', header=0, skiprows=list(range(1,58)), engine='c')#usecols = [0,1,2])

# halos = halos[halos['pid(5)']==-1]

fig1, axes = plt.subplots(1,2) #, figsize=(9,7.5))#, dpi=120)

mass_unit = r'$h^{-1}M_{\odot}$'

# bins = np.logspace(-3,4)
# axes[0,0].hist(halos['Acc_Rate_1*Tdyn(65)'], log=True, bins=bins)
# axes[0,0].set_xlabel(mass_unit+ '/yr')
# axes[0,0].set_xscale('log')

# axes[0,1].scatter(halos['mvir(10)'], halos['Acc_Rate_1*Tdyn(64)'])
# axes[0,1].set_xlabel(mass_unit)
# axes[0,1].set_xscale('log')
# axes[0,1].set_xlabel(mass_unit+ '/yr')
# axes[0,1].set_yscale('log')

# halos.hist('Acc_Rate_2*Tdyn(65)', ax=axes[0,1], loglog=True, xlabel=mass_unit+'/yr')

halos.plot.scatter('mvir(10)', 'Acc_Rate_2*Tdyn(65)', loglog=True, ylim=(1e0,1e6), ax=axes[0], xlabel=mass_unit, ylabel=mass_unit+'/yr', s=1)

# plt.tight_layout()
fig1.savefig(f'accretion_rate_snap_{i}.svg')
fig1.savefig(f'accretion_rate_snap_{i}.png')


