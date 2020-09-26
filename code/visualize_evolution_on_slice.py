import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fig1, ax1 = plt.subplots(dpi=100)
im1 = ax1.imshow(delta_slice+1, extent=[0,snap.box_size,0,snap.box_size], cmap='inferno', norm=LogNorm())
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
ax1.set_title("Density field assigned from the particle positions in the snapshot")
ax1.set_xlabel("$h^{-1}$Mpc")
ax1.set_ylabel("$h^{-1}$Mpc")
# ax1.set_xscale('log')


fig1.savefig('delta2d.pdf', bbox_inchex='tight')