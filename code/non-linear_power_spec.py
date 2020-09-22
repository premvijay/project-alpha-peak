import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from library.gadget_tools import Snapshot
from library.pm_tools import assign_density
from library.field_tools import compute_power_spec
# from . import library
# import library

import os
import sys

# print(os.getcwd())

binary_files_dir = '/media/premv/Seagate Backup Plus Drive/work_large_files/'
binary_files_dir =  '/mnt/scratch/aseem/sims/bdm_cdm1024/r1/'
# binary_files_dir = sys.argv[1]
print(binary_files_dir)

snapshot_number = 200
# snapshot_number = sys.argv[2]


pos_list = []

file_number = 0
while True:
    filename = '/snapshot_{0:03d}.{1}'.format(snapshot_number,file_number)
    # filepath = os.path.join(binary_files_dir, filename)
    filepath = binary_files_dir + filename
    print(filepath)

    snap = Snapshot()
    snap.from_binary(filepath)
    pos_list.append(snap.positions(prtcl_type="Halo", max_prtcl=1000))

    if file_number == snap.num_files-1:
        break
    else:
        file_number += 1

posd = np.vstack(pos_list)

delta, raw_grid = assign_density(posd, snap.box_size, grid_size = 512, scheme='CIC')


# print(posd)



# log10_delta_plus_1 = np.log10(1 + delta)
delta_plus_1 = (1 + delta)
# delta_plus_1 = raw_grid

# print(raw_grid)

print(delta_plus_1.shape)
# print((delta_plus_1))






fig1, ax1 = plt.subplots(dpi=100)
im1 = ax1.imshow(delta_plus_1[:,:,0],extent=[0,snap.box_size,0,snap.box_size], cmap='inferno', norm=LogNorm(vmin=1e-1))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title("Density field assigned from the particle positions in the snapshot")
ax1.set_xlabel("$h^{-1}$Mpc")
ax1.set_ylabel("$h^{-1}$Mpc")
# ax1.set_xscale('log')

fig1.savefig('delta2d.pdf', bbox_inchex='tight')



power_spec = compute_power_spec(delta,snap.box_size)
power_spec.to_csv('k,Pk.csv')

fig2, ax2 = plt.subplots(2,1)
ax2[0].plot(power_spec['k'],power_spec['Pk'])
# ax2[1].plot(power_spec['lam'],power_spec['Pk'])

fig2.savefig('power_spectrum.pdf', bbox_inchex='tight')





