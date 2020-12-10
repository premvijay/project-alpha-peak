import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation
import pandas as pd
import argparse
import pdb

from gadget_tools import Snapshot


parser = argparse.ArgumentParser(
    description='Density field evolution from Gadget simulation.',
    usage= 'python ./visualize_simulation.py')

parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--plots_into', type=str, default='/mnt/home/student/cprem/project-alpha-peak/notes_and_results/plots_and_anims')

parser.add_argument('--snap_i', type=int, default=200, help='Snapshot index number')

parser.add_argument('--light_snaps', type=int, default=1, help='save white bg images for pdf notes')

args = parser.parse_args()

grid_size = 512
scheme = 'TSC'
# rundir = 'r1'
interlaced = True

inlcd_str = '_interlaced' if interlaced==True else ''

if args.light_snaps:
    theme = '_light'
    plt.style.use('default')
    # plt.set_cmap('nipy_spectral')
else:
    theme = ''
    plt.style.use('dark_background')
    # plt.set_cmap('nipy_spectral')

# simname = 'bdm_cdm1024' if args.simname is None else args.simname

schemes = ['NGP', 'CIC', 'TSC']

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', f'full_box')
os.makedirs(plotsdir, exist_ok=True)

i = args.snap_i

simdir = os.path.join('/scratch/aseem/sims', args.simname, args.rundir)
snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

box_size = snap.box_size

fig1, ax2 = plt.subplots(1, figsize=(7.5,7), dpi=150)

ax2.plot([],[], ' ', label=f"Snapshot-{i:03d}, Grid-size: {grid_size:d}")

# fig1.suptitle("Simulation: {2}, ".format(i,snap.redshift,args.simname,grid_size)    )

for scheme in schemes:
    savesdir = os.path.join('/scratch/cprem/sims', args.simname, args.rundir, 'global', scheme, '{0:d}'.format(grid_size))
    print(savesdir)

    Pkdir = os.path.join(savesdir,'power_spectrum')
    # infodir = os.path.join(savesdir,'info')

    # ax1.set_xscale('log')
    
    power_spec = pd.read_csv(os.path.join(Pkdir, 'Pk_{0:03d}.csv'.format(i)), sep='\t', dtype='float64')
    power_spec.columns = ['k', 'Pk']

    if interlaced:
        power_spec_inlcd = pd.read_csv(os.path.join(Pkdir, 'Pk{1}_{0:03d}.csv'.format(i,inlcd_str)), sep='\t', dtype='float64')
        power_spec_inlcd.columns = ['k', 'Pk']

    lin_bin = np.linspace(power_spec['k'].iloc[1],power_spec['k'].iloc[-10], 200)
    log_bin = np.logspace(-2,1.3, 100)
    merge_bin = np.concatenate([lin_bin,log_bin])
    merge_bin.sort()


    power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=lin_bin)).mean()
    ax2.plot(power_spec_grouped1['k'],power_spec_grouped1['Pk'], label=f"{scheme:s} scheme without interlacing")[0]

    if interlaced:
        power_spec_inlcd_grouped1 = power_spec_inlcd.groupby(pd.cut(power_spec_inlcd['k'], bins=lin_bin)).mean()
        ax2.plot(power_spec_inlcd_grouped1['k'],power_spec_inlcd_grouped1['Pk'], label=f"{scheme:s} scheme with interlacing")[0]



# pdb.set_trace()


# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k$  ($h$ Mpc$^{-1}$)")
ax2.set_ylabel(r"$P(k)$  ($h^{-1}$Mpc)$^3$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(top=1e5)
ax2.grid(True)
ax2.set_title(f"Matter power spectrum from {args.simname:s} simulation at redshift z={f'{snap.redshift:.3f}'.rstrip('0').rstrip('.'):s} ")
ax2.legend()

plt.tight_layout()

fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_vary_scheme_{i:03d}{theme:s}.pdf'))
fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_vary_scheme_{i:03d}{theme:s}.png'))
# fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_{i:03d}{theme:s}.svg'))