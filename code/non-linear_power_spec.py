import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os
import sys
from time import time

from gadget_tools import Snapshot
from pm_tools import assign_density, project_to_slice
from field_tools import compute_power_spec

# print(os.getcwd())

grid_size = 128



# binary_files_dir = '/media/premv/Seagate Backup Plus Drive/work_large_files/'
binary_files_dir =  '/scratch/aseem/sims/bdm_cdm1024/r1/'
# binary_files_dir = sys.argv[1]
print(binary_files_dir)

snapshot_number = 0
# snapshot_number = sys.argv[2]

t_now = time()
print('\n Starting to read snapshots binaries')


pos_list = []

file_number = 0
while True:
    filename = '/snapshot_{0:03d}.{1}'.format(snapshot_number,file_number)
    # filepath = os.path.join(binary_files_dir, filename)
    filepath = binary_files_dir + filename
    print(filepath)

    snap = Snapshot()
    snap.from_binary(filepath)
    # if posd is None:
    #     posd = snap.positions(prtcl_type="Halo", max_prtcl=None)
    # else:
    #     posd = np.vstack([posd,snap.positions(prtcl_type="Halo", max_prtcl=None)])
    pos_list.append(snap.positions(prtcl_type="Halo", max_prtcl=None))
    t_bef, t_now = t_now, time()
    print(t_now-t_bef)
    if file_number == snap.num_files-1:
        break
    else:
        file_number += 1

print('\n Particle positions read from all binaries in the snapshot')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

posd = np.vstack(pos_list)
del pos_list[:]

print('\n particle positions stacked')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

delta = assign_density(posd, snap.box_size, grid_size, scheme='CIC')




print('\n density assigned')
t_bef, t_now = t_now, time()
print(t_now-t_bef)
print('\n Sizes of pos_list is {0}, posd is {1}, delta is {2}'.format(sys.getsizeof(pos_list),
                 posd.nbytes, delta.nbytes))

del posd


# log10_delta_plus_1 = np.log10(1 + delta)
# delta_plus_1 = (1 + delta)

# print(delta_plus_1.shape)
# print((delta_plus_1))

print('\n delta_plus_1 obtained')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

mmhpos = (48.25266, 166.29897, 98.36325)

delta_slice = project_to_slice(delta, snap.box_size, axis=2, around_position=mmhpos, thick=10)


fig1, ax1 = plt.subplots(dpi=100)
im1 = ax1.imshow(delta_slice+1, extent=[0,snap.box_size,0,snap.box_size], cmap='inferno', norm=LogNorm(vmin=1e-1))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title("Density field assigned from the particle positions in the snapshot")
ax1.set_xlabel("$h^{-1}$Mpc")
ax1.set_ylabel("$h^{-1}$Mpc")
# ax1.set_xscale('log')

fig1.savefig('delta2d.pdf', bbox_inchex='tight')

print('\n 2d slice plotted with imshow and saved')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

power_spec = compute_power_spec(delta,snap.box_size)

print('\n power spectrum computed')
t_bef, t_now = t_now, time()
print(t_now-t_bef)



power_spec.to_csv('power_spec_{0:03d}.csv'.format(snapshot_number), sep='\t', index=False, 
                            float_format='%.8e', header=['k (h/Mpc)', 'P(k) (Mpc/h)^3'])

print('\n power spectrum saved to csv')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

fig2, ax2 = plt.subplots(1,1)
ax2.plot(power_spec['k'],power_spec['Pk'])
# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k ~~(h~\mathrm{Mpc}^{-1}$)")
ax2.set_ylabel(r"$P(k) ~~(h^{-1}\mathrm{Mpc})^3$")


ax2.set_xscale('log')
ax2.set_yscale('log')

fig2.savefig('power_spec_{0:03d}.pdf'.format(snapshot_number))

print('\n power spectrum plotted and saved')
t_bef, t_now = t_now, time()
print(t_now-t_bef)



