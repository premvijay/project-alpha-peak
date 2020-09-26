import matplotlib.pyplot as plt




fig2, ax2 = plt.subplots(1,1)
ax2.plot(power_spec['k'],power_spec['Pk'])
# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k ~~(h~\mathrm{Mpc}^{-1}$)")
ax2.set_ylabel(r"$P(k) ~~(h^{-1}\mathrm{Mpc})^3$")


ax2.set_xscale('log')
ax2.set_yscale('log')

fig2.savefig('power_spec_{0:03d}.pdf'.format(snapshot_number))