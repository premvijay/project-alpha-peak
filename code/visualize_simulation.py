import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation
import pandas as pd
import argparse
import pdb

parser = argparse.ArgumentParser(
    description='Density field evolution from Gadget simulation.',
    usage= 'python ./visualize_evolution_on_slice.py')

parser.add_argument('--simname', type=str, help='Directory name containing the saved data')

args = parser.parse_args()


# if args.slicedir is None:
#     if args.simname is None:
#         slicedir = '/scratch/cprem/sims/bdm_cdm1024/slice2D/'
#     else:
#         slicedir = '/scratch/cprem/sims/' + simname + '/slice2D/'
# else:
#     slicedir = args.slicedir

simdir = '/scratch/cprem/sims/' + args.simname + '/'

print(simdir)


box_size =200

i = 150

delta_slice = np.load(simdir+'/slice2D/slice_{0:03d}.npy'.format(i))


fig1, (ax1,ax2) = plt.subplots(1,2, figsize=(13,7), dpi=150)

im1 = ax1.imshow(delta_slice+1, extent=[0,box_size,0,box_size], cmap='inferno', norm=LogNorm())
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title(r"10 $h^{-1}$ Mpc thick slice")
ax1.set_xlabel(r"$h^{-1}$Mpc")
ax1.set_ylabel(r"$h^{-1}$Mpc")
# ax1.set_xscale('log')

power_spec = pd.read_csv(simdir + '/power_spectrum/Pk_{0:03d}.csv'.format(i), sep='\t', dtype='float64')
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


fig1.savefig(simdir+'/plots_and_anims/delta2d.pdf', bbox_inchex='tight')

def update(i):
    print(i, 'starting')
    delta_slice = np.load(simdir+'/slice2D/slice_{0:03d}.npy'.format(i))
    im1.set_data(delta_slice+1)
    
    power_spec = pd.read_csv(simdir + '/power_spectrum/Pk_{0:03d}.csv'.format(i), sep='\t', dtype='float64')
    power_spec.columns = ['k', 'Pk']
    power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=lin_bin)).mean()
    # print(power_spec_grouped1['k'].values)
    plot2.set_xdata(power_spec_grouped1['k'].values)
    plot2.set_ydata(power_spec_grouped1['Pk'].values)
    # ax2.clear()
    # plot2 = ax2.plot(power_spec_grouped1['k'],power_spec_grouped1['Pk'])[0]
    fig1.suptitle("Snapshot-{} at redshift z=".format(i))
    # print(power_spec_grouped1)
    print(i,'stopping')


anim = matplotlib.animation.FuncAnimation(fig1, update, frames=201, interval=200)

# plt.rcParams['animation.ffmpeg_path'] = ''

# Writer=matplotlib.animation.ImageMagickWriter
Writer=matplotlib.animation.FFMpegWriter
writer = Writer(fps=15)

anim.save(simdir+"/plots_and_anims/simulation_visualisation.mp4", writer=writer, dpi=150)
print("saved")