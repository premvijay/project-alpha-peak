import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from library.gadget_tools import Snapshot
from library.pm_tools import assign_density
# from . import library
# import library

import os
import sys

print(os.getcwd())

binary_files_dir = '/media/premv/Seagate Backup Plus Drive/work_large_files/'

filepath = binary_files_dir + '/snapshot_200.0'

snap = Snapshot()
snap.from_binary(filepath)

posd = snap.positions(prtcl_type="Halo", max_prtcl=40000000)

print(posd)

delta = assign_density(posd, snap.box_size, grid_size = 256)

# log10_delta_plus_1 = np.log10(1 + delta)
delta_plus_1 = (1 + delta)

print(delta_plus_1.shape)
print((delta_plus_1))


fig1, ax1 = plt.subplots(dpi=100)
im1 = ax1.imshow(delta_plus_1[:,:,0],extent=[0,snap.box_size,0,snap.box_size], cmap='inferno', norm=LogNorm(vmin=1))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title("Density field assigned from the particles in the snapshot")
ax1.set_xlabel("$h^{-1}$Mpc")
ax1.set_ylabel("$h^{-1}$Mpc")
# ax1.set_xscale('log')
plt.show()