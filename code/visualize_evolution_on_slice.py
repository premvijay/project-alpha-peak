import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation
import argparse
import pdb

parser = argparse.ArgumentParser(
    description='Density field evolution from Gadgat simulation.',
    usage= 'python ./visualize_evolution_on_slice.py --slicedir /scratch/cprem/sims/bdm_cdm1024/slice2D/')

parser.add_argument('--slicedir', type=str, help='Directory path containing the saved slice numpy')
parser.add_argument('--simname', type=str, help='Directory name containing the saved slice numpy')

args = parser.parse_args()


if args.slicedir is None:
    if args.simname is None:
        slicedir = '/scratch/cprem/sims/bdm_cdm1024/slice2D/'
    else:
        slicedir = '/scratch/cprem/sims/' + simname + '/slice2D/'
else:
    slicedir = args.slicedir

print(slicedir)

box_size =200

i = 150

delta_slice = np.load(slicedir+'/slice_{0:03d}.npy'.format(i))


fig1, ax1 = plt.subplots(dpi=100)
im1 = ax1.imshow(delta_slice+1, extent=[0,box_size,0,box_size], cmap='inferno', norm=LogNorm())
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title("2D slice of density field from the {:d}th snapshot".format(i))
ax1.set_xlabel("$h^{-1}$Mpc")
ax1.set_ylabel("$h^{-1}$Mpc")
# ax1.set_xscale('log')


fig1.savefig(slicedir+'delta2d.pdf', bbox_inchex='tight')

def update2D(i):
    print(i, 'starting')
    delta_slice = np.load(slicedir+'/slice_{0:03d}.npy'.format(i))
    im1.set_data(delta_slice+1)
    ax1.set_title("2D slice of density field from the {:d}th snapshot".format(i))


anim = matplotlib.animation.FuncAnimation(fig1, update2D, frames=201, interval=200)

anim.save(slicedir+"/density_evolution_on_slice.gif", writer="imagemagick")
print("saved")
