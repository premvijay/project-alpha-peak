import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import copy
import argparse
import pdb
from time import time

from gadget_tools import Snapshot


import sys
assert sys.version_info[0] == 3, "Must be using Python 3"

plt.style.use('dark_background')

parser = argparse.ArgumentParser(description='Find most massive in a given tree.',
 usage= 'python')

# parser.add_argument('--treefile', type=str)
parser.add_argument('--simsdir', type=str, default='/scratch/aseem/sims', help='Directory path for sims saved data')
parser.add_argument('--halosdir', type=str, default='/scratch/aseem/halos', help='Directory path for halos saved data')
parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--plots_into', type=str, default='/mnt/home/student/cprem/project-alpha-peak/notes_and_results/plots_and_anims')


args = parser.parse_args()


treesdir = os.path.join(args.halosdir, args.simname, args.rundir)
simdir = os.path.join(args.simsdir, args.simname, args.rundir)

outdir = os.path.join('/scratch/cprem/sims',args.simname,args.rundir,'halo_centric','halos_list')
os.makedirs(outdir, exist_ok=True)
print(outdir)

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', 'accretion_rate')
# print('hi')
os.makedirs(plotsdir, exist_ok=True)

i = 200
treefile = os.path.join(treesdir, 'out_{0:03d}.trees'.format(i))
snapfile = os.path.join(simdir, 'snapshot_{0:03d}.0'.format(i))
print(treefile)
snap = Snapshot(snapfile=snapfile)
print(snap.Hubble_param)

t_H = 9.778e9 / snap.Hubble_param

halos = pd.read_csv(treefile, sep=r'\s+', header=0, skiprows=list(range(1,58)), engine='c')#usecols = [0,1,2])

# halos = halos[halos['pid(5)']==-1]

fig1, axes = plt.subplots(1,2 , figsize=(15,7.5))#, dpi=120)

mass_unit = r'$h^{-1}M_{\odot}$'

bins = np.logspace(-7,3)
axes[0].hist(halos['Acc_Rate_2*Tdyn(65)']/halos['mvir(10)']*t_H, log=True, density=True, bins=bins)
# axes[0].set_xlabel(mass_unit+ '/yr')
# axes[0].set_ylabel('Number of halos')
axes[0].set_xlabel(r'Specific accretion rate $ \times ~ t_H$')
axes[0].set_xscale('log')
axes[0].set_xlim(1e-7,1e3)
axes[0].set_title(f'Histogram of specific accretion rate at redshift, z={round(snap.redshift,4):.4g}')

axes[1].scatter(halos['Acc_Rate_2*Tdyn(65)']/halos['mvir(10)']*t_H, halos['mvir(10)'],  s=.25, alpha=.3)
axes[1].set_ylabel('virial mass '+mass_unit)
axes[1].set_yscale('log')
# axes[1].set_xlabel(mass_unit+ '/yr')
axes[1].set_xlabel(r'Specific accretion rate $ \times ~ t_H$')
axes[1].set_xscale('log')
axes[1].set_xlim(1e-7,1e3)

# halos.hist('Acc_Rate_2*Tdyn(65)', ax=axes[1], loglog=True, xlabel=mass_unit+'/yr')

# halos.plot.scatter('mvir(10)', 'Acc_Rate_2*Tdyn(65)', loglog=True, ylim=(1e0,1e6), ax=axes[0], xlabel=mass_unit, ylabel=mass_unit+'/yr', s=1)

plt.tight_layout()
plt.rc('font', size=12) 
# fig1.savefig(os.path.join(plotsdir, f'snap_{i}.svg'))
fig1.savefig(os.path.join(plotsdir, f'snap_{i}_2tdyn.png'))


