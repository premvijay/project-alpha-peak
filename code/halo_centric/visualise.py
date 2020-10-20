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

parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--downsample', type=int, default=8, 
                help='visualize the Downsampled particles in simulation by this many times')

parser.add_argument('--tree_root', type=int, default=200)
parser.add_argument('--M_around', type=float, default=3e12)
parser.add_argument('--max_halos', type=int, default=500)

parser.add_argument('--align', action='store_true', help='Visualize aligned and then stacked images')

parser.add_argument('--outdir', type=str, default='/scratch/cprem/sims')

args = parser.parse_args()

grid_size = 512
scheme = 'TSC'


# simname = 'bdm_cdm1024' if args.simname is None else args.simname

savesdir_global = os.path.join(args.outdir, args.simname, args.rundir, 'TSC', '512')

savesdir = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', scheme, '{0:d}'.format(grid_size))

print(savesdir)

halosfile = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', 'halos_list',
'halos_select_{0:.1e}_{1:d}.csv'.format(args.M_around,args.max_halos))

slicedir = os.path.join(savesdir,'slice2D')
infodir = os.path.join(savesdir_global,'info')
plotsdir = os.path.join(savesdir, 'plots_and_anims')
os.makedirs(plotsdir, exist_ok=True)

align_str = ''
if not args.align:
    slicedir = os.path.join(slicedir, 'unaligned')
    align_str += '_unaligned'

halos = pd.read_csv(halosfile, engine='c', index_col='id(1)')
halos_root = halos[halos['Snap_num(31)']==args.tree_root]

i = 150

halos_this_step = halos[halos['Snap_num(31)']==i]


with open(os.path.join(infodir, 'header_{0:03d}.p'.format(i)), 'rb') as infofile:
    snap=pickle.load(infofile)

mean_dens_comoving = np.dot(snap.mass_table*1e10, snap.N_prtcl_total) / snap.box_size**3

with open( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}_{2:.1e}_{3:d}.meta'.format(i, args.downsample, args.M_around,args.max_halos)), 'rt' ) as metafile:
    metadict = json.load(metafile)

box_size = metadict['L_cube']



delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}_{2:.1e}_{3:d}.npy'.format(i, args.downsample, args.M_around,args.max_halos)) )


fig1, ax1 = plt.subplots(figsize=(9,7), dpi=150)


fig1.suptitle("Snapshot-{0:03d} at redshift z={1:.4f};     Simulation: {2}, Grid size: {3}, Scheme: {4}".format(i,snap.redshift,args.simname,grid_size,scheme))

im1 = ax1.imshow(delta_slice+1+1e-5, extent=[-box_size/2,box_size/2,-box_size/2,box_size/2], cmap='nipy_spectral', norm=LogNorm(vmin=3e-1))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title(r"0.25 $h^{-1}$ Mpc thick slice in halo centric stack of "+'{}'.format(metadict['N_stack']))
ax1.set_xlabel(r"$h^{-1}$Mpc")
ax1.set_ylabel(r"$h^{-1}$Mpc")

r_vir_circ = plt.Circle((0,0),radius=metadict['R_vir'], fill=False, label='virial radius from simulation')
ax1.add_patch(r_vir_circ)

# M_vir = args.M_around 
M_vir = halos_this_step['mvir(10)'].mean()
R_vir_sc = ( M_vir / (4/3 * np.pi * 178 * mean_dens_comoving) )**(1/3)
r_vir_sc_circ = plt.Circle((0,0),radius=R_vir_sc, fill=False, ls='--', label='virial radius from spherical collapse')
ax1.add_patch(r_vir_sc_circ)

plt.legend()
# ax1.set_xscale('log')




fig1.savefig(os.path.join(plotsdir, 'single_snapshot{4}_{0:03d}_1by{1:d}_{2:.1e}_{3:d}.pdf'.format(i, args.downsample, args.M_around,args.max_halos, align_str)), bbox_inchex='tight')

def update(i):
    print(i, 'starting')
    delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}_{2:.1e}_{3:d}.npy'.format(i, args.downsample, args.M_around,args.max_halos)) )
    im1.set_data(delta_slice+1+1e-5)
    
    with open( os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}_{2:.1e}_{3:d}.meta'.format(i, args.downsample, args.M_around,args.max_halos)), 'rt' ) as metafile:
        metadict = json.load(metafile)
    halos_this_step = halos[halos['Snap_num(31)']==i]
    M_vir = halos_this_step['mvir(10)'].mean()
    R_vir_sc = ( M_vir / (4/3 * np.pi * 178 * mean_dens_comoving) )**(1/3)

    r_vir_circ.set_radius(metadict['R_vir'])
    r_vir_sc_circ.set_radius(R_vir_sc)

    ax1.set_title(r"0.25 $h^{-1}$ Mpc thick slice in halo centric stack of "+'{}'.format(metadict['N_stack']))

    with open(os.path.join(infodir, 'header_{0:03d}.p'.format(i)), 'rb') as infofile:
        snap=pickle.load(infofile)
    fig1.suptitle("Snapshot-{0:03d} at redshift z={1:.4f};     Simulation: {2}, Grid size: {3}, Scheme: {4}".format(i,snap.redshift,args.simname,grid_size,scheme))
    print(i,'stopping')


anim = matplotlib.animation.FuncAnimation(fig1, update, frames=range(6,args.tree_root+1), interval=500)

# plt.rcParams['animation.ffmpeg_path'] = ''

# Writer=matplotlib.animation.ImageMagickWriter
Writer=matplotlib.animation.FFMpegWriter
writer = Writer(fps=10)

anim.save(os.path.join(plotsdir, 'simulation_visualisation{3}_1by{0:d}_{1:.1e}_{2:d}.mp4'.format(args.downsample, args.M_around,args.max_halos, align_str)), writer=writer, dpi=150)
print("saved")