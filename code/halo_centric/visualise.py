import os
import pickle, json
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

parser.add_argument('--simname', type=str, help='Directory name containing the saved data')

parser.add_argument('--downsample', type=int, default=8, 
                help='visualize the Downsampled particles in simulation by this many times')

args = parser.parse_args()

grid_size = 512
scheme = 'TSC'
rundir = 'r1'


simname = 'bdm_cdm1024' if args.simname is None else args.simname

savesdir_global = os.path.join('/scratch/cprem/sims', simname, rundir, 'TSC', '512')

savesdir = os.path.join('/scratch/cprem/sims', simname, rundir, 'halo_centric', scheme, '{0:d}'.format(grid_size))

print(savesdir)


slicedir = os.path.join(savesdir,'slice2D')
infodir = os.path.join(savesdir_global,'info')
plotsdir = os.path.join(savesdir, 'plots_and_anims')
os.makedirs(plotsdir, exist_ok=True)

i = 150

with open(os.path.join(infodir, 'header_{0:03d}.p'.format(i)), 'rb') as infofile:
    snap=pickle.load(infofile)

with open( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}.meta'.format(i, args.downsample)), 'rt' ) as metafile:
    metadict = json.load(metafile)

box_size = metadict['L_cube']



delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}.npy'.format(i, args.downsample)) )


fig1, ax1 = plt.subplots(figsize=(9,7), dpi=150)

fig1.suptitle("Snapshot-{0:03d} at redshift z={1:.4f};     Simulation: {2}, Grid size: {3}, Scheme: {4}".format(i,snap.redshift,simname,grid_size,scheme))

im1 = ax1.imshow(delta_slice+1+1e-5, extent=[-box_size/2,box_size/2,-box_size/2,box_size/2], cmap='nipy_spectral', norm=LogNorm(vmin=3e-1))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title(r"0.25 $h^{-1}$ Mpc thick slice in halo centric stack of "+'{}'.format(metadict['N_stack']))
ax1.set_xlabel(r"$h^{-1}$Mpc")
ax1.set_ylabel(r"$h^{-1}$Mpc")
# ax1.set_xscale('log')




fig1.savefig(os.path.join(plotsdir, 'single_snapshot_{0:03d}_1by{1:d}.pdf'.format(i, args.downsample)), bbox_inchex='tight')

def update(i):
    print(i, 'starting')
    delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}.npy'.format(i, args.downsample)) )
    im1.set_data(delta_slice+1+1e-5)
    
    with open( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}.meta'.format(i, args.downsample)), 'rt' ) as metafile:
        metadict = json.load(metafile)
    ax1.set_title(r"0.25 $h^{-1}$ Mpc thick slice in halo centric stack of "+'{}'.format(metadict['N_stack']))

    with open(os.path.join(infodir, 'header_{0:03d}.p'.format(i)), 'rb') as infofile:
        snap=pickle.load(infofile)
    fig1.suptitle("Snapshot-{0:03d} at redshift z={1:.4f};     Simulation: {2}, Grid size: {3}, Scheme: {4}".format(i,snap.redshift,simname,grid_size,scheme))
    print(i,'stopping')


anim = matplotlib.animation.FuncAnimation(fig1, update, frames=range(1,201), interval=500)

# plt.rcParams['animation.ffmpeg_path'] = ''

# Writer=matplotlib.animation.ImageMagickWriter
Writer=matplotlib.animation.FFMpegWriter
writer = Writer(fps=10)

anim.save(os.path.join(plotsdir, 'simulation_visualisation_1by{0:d}.mp4'.format(args.downsample)), writer=writer, dpi=150)
print("saved")