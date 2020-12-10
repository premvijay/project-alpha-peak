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

parser.add_argument('--snap_i', type=int, default=150, help='Snapshot index number')

parser.add_argument('--light_snaps', type=int, default=1, help='save white bg images for pdf notes')

parser.add_argument('--scheme', type=str, help='Scheme for assigning particles to grid')

args = parser.parse_args()

grid_size = 512
scheme = args.scheme
# rundir = 'r1'
# interlaced = True

if args.light_snaps:
    theme = '_light'
    plt.style.use('default')
    # plt.set_cmap('nipy_spectral')
else:
    theme = ''
    plt.style.use('dark_background')
    # plt.set_cmap('nipy_spectral')

# simname = 'bdm_cdm1024' if args.simname is None else args.simname

savesdir = os.path.join('/scratch/cprem/sims', args.simname, args.rundir, 'global', scheme, '{0:d}'.format(grid_size))

print(savesdir)

simdir = os.path.join('/scratch/aseem/sims', args.simname, args.rundir)

slicedir = os.path.join(savesdir,'slice2D')
Pkdir = os.path.join(savesdir,'power_spectrum')
infodir = os.path.join(savesdir,'info')

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', f'full_box_{scheme:s}_{grid_size:d}')
os.makedirs(plotsdir, exist_ok=True)

i = args.snap_i

snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

box_size = snap.box_size

delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}.npy'.format(i)) )

fig1, ax1 = plt.subplots(1, figsize=(7.5,6.5), dpi=150)

if not args.light_snaps:
    fig1.suptitle("Simulation: {2}, Grid size: {3}, Scheme: {4} \n Snapshot-{0:03d} at redshift z={1:.3f}".format(i,snap.redshift,args.simname,grid_size,scheme)    )

im1 = ax1.imshow(delta_slice+1, extent=[0,box_size,0,box_size], cmap='inferno', norm=LogNorm(vmin=5e-2,vmax=5e2))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title(r"10 $h^{-1}$ Mpc thick slice"+ f" at redshift z={f'{snap.redshift:.3f}'.rstrip('0').rstrip('.'):s} ({i:d}th snapshot)")
ax1.set_xlabel(r"$h^{-1}$Mpc")
ax1.set_ylabel(r"$h^{-1}$Mpc")
# ax1.set_xscale('log')
# pdb.set_trace()

plt.tight_layout()

fig1.savefig(os.path.join(plotsdir, f'single_snapshot_image_{i:03d}{theme:s}.pdf'))
fig1.savefig(os.path.join(plotsdir, f'single_snapshot_image_{i:03d}{theme:s}.png'))
# fig1.savefig(os.path.join(plotsdir, f'single_snapshot_image_{i:03d}{theme:s}.svg'))