import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation
import argparse
import pdb

parser = argparse.ArgumentParser(
    description='Density field evolution from Gadgat simulation.',
    usage= 'python ./visualize_evolution_on_slice.py --slicedir ~/myscratch/bdm_cdm1024/slice2D/')

parser.add_argument('--slicedir', type=str, help='Directory containing the saved power spectrum csv')


args = parser.parse_args()
box_size =200

i = 150

delta_slice = np.load(args.slicedir+'/slice_{0:03d}.npy'.format(i))


fig1, ax1 = plt.subplots(dpi=100)
im1 = ax1.imshow(delta_slice+1, extent=[0,box_size,0,box_size], cmap='inferno', norm=LogNorm())
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title("2D slice of density field from the {:d}th snapshot".format(i))
ax1.set_xlabel("$h^{-1}$Mpc")
ax1.set_ylabel("$h^{-1}$Mpc")
# ax1.set_xscale('log')


fig1.savefig(args.slicedir+'delta2d.pdf', bbox_inchex='tight')

def update2D(i):
    delta_slice = np.load(args.slicedir+'/slice_{0:03d}.npy'.format(i))
    im1.set_data(delta_slice+1)
    ax1.set_title("2D slice of density field from the {:d}th snapshot".format(i))


anim = matplotlib.animation.FuncAnimation(fig1, update2D, frames=201, interval=200)

anim.save(args.slicedir+"/density_evolution_on_slice.mp4")#, writer="imagemagick")
print("saved")
