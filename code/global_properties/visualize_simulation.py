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

plt.style.use('dark_background')

parser = argparse.ArgumentParser(
    description='Density field evolution from Gadget simulation.',
    usage= 'python ./visualize_simulation.py')

parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--plots_into', type=str, default='/mnt/home/student/cprem/project-alpha-peak/notes_and_results/plots_and_anims')

args = parser.parse_args()

grid_size = 512
scheme = 'TSC'
# rundir = 'r1'
interlaced = True

inlcd_str = '_interlaced' if interlaced==True else ''

# simname = 'bdm_cdm1024' if args.simname is None else args.simname

savesdir = os.path.join('/scratch/cprem/sims', args.simname, args.rundir, scheme, '{0:d}'.format(grid_size))

print(savesdir)


slicedir = os.path.join(savesdir,'slice2D')
Pkdir = os.path.join(savesdir,'power_spectrum')
infodir = os.path.join(savesdir,'info')
plotsdir = os.path.join(savesdir, 'plots_and_anims')
os.makedirs(plotsdir, exist_ok=True)

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', f'full_box_{scheme:s}_{grid_size:d}')
os.makedirs(plotsdir, exist_ok=True)

i = 150

with open(os.path.join(infodir, 'header_{0:03d}.p'.format(i)), 'rb') as infofile:
    snap=pickle.load(infofile)



box_size = snap.box_size



delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}.npy'.format(i)) )


fig1, (ax1,ax2) = plt.subplots(1,2, figsize=(13,7), dpi=150)

fig1.suptitle("Snapshot-{0:03d} at redshift z={1:.4f};     Simulation: {2}, Grid size: {3}, Scheme: {4}".format(i,snap.redshift,args.simname,grid_size,scheme))

im1 = ax1.imshow(delta_slice+1, extent=[0,box_size,0,box_size], cmap='inferno', norm=LogNorm())
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title(r"10 $h^{-1}$ Mpc thick slice")
ax1.set_xlabel(r"$h^{-1}$Mpc")
ax1.set_ylabel(r"$h^{-1}$Mpc")
# ax1.set_xscale('log')

power_spec = pd.read_csv(os.path.join(Pkdir, 'Pk{1}_{0:03d}.csv'.format(i,inlcd_str)), sep='\t', dtype='float64')
power_spec.columns = ['k', 'Pk']

lin_bin = np.linspace(power_spec['k'].iloc[1],power_spec['k'].iloc[-10], 200)
log_bin = np.logspace(-2,1.3, 300)


power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=lin_bin)).mean()
# pdb.set_trace()

plot2 = ax2.plot(power_spec_grouped1['k'],power_spec_grouped1['Pk'])[0]
# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k$  ($h$ Mpc$^{-1}$)")
ax2.set_ylabel(r"$P(k)$  ($h^{-1}$Mpc)$^3$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(top=1e5)
ax2.grid(True)
ax2.set_title("Power spectrum")


plt.tight_layout()

fig1.savefig(os.path.join(plotsdir, 'single_snapshot.pdf'))
fig1.savefig(os.path.join(plotsdir, 'single_snapshot.png'))
fig1.savefig(os.path.join(plotsdir, 'single_snapshot.svg'))

def update(i):
    print(i, 'starting')
    delta_slice = np.load( os.path.join(slicedir, 'slice_{0:03d}.npy'.format(i)) )
    im1.set_data(delta_slice+1)
    
    power_spec = pd.read_csv(os.path.join(Pkdir, 'Pk{1}_{0:03d}.csv'.format(i,inlcd_str)), sep='\t', dtype='float64')
    power_spec.columns = ['k', 'Pk']
    power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=lin_bin)).mean()
    # print(power_spec_grouped1['k'].values)
    plot2.set_xdata(power_spec_grouped1['k'].values)
    plot2.set_ydata(power_spec_grouped1['Pk'].values)

    with open(os.path.join(infodir, 'header_{0:03d}.p'.format(i)), 'rb') as infofile:
        snap=pickle.load(infofile)
    fig1.suptitle("Snapshot-{0:03d} at redshift z={1:.4f};     Simulation: {2}, Grid size: {3}, Scheme: {4}".format(i,snap.redshift,args.simname,grid_size,scheme))
    print(i,'stopping')


anim = matplotlib.animation.FuncAnimation(fig1, update, frames=201, interval=500)

# plt.rcParams['animation.ffmpeg_path'] = ''

# Writer=matplotlib.animation.ImageMagickWriter
Writer=matplotlib.animation.FFMpegWriter
writer = Writer(fps=10)

anim.save(os.path.join(plotsdir, 'simulation_visualisation.mp4'), writer=writer, dpi=150)
print("saved")