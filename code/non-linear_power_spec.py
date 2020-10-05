import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd
import argparse
import pdb

parser = argparse.ArgumentParser(
    description='Animation of power spectrum.',
    usage= 'python ./non-linear_power_spec.py --Pkdir ~/myscratch/bdm_cdm1024/power_spectrum/')

parser.add_argument('--Pkdir', type=str, help='Directory containing the saved power spectrum csv')


args = parser.parse_args()

i = 0
power_spec = pd.read_csv(args.Pkdir+'/Pk_{0:03d}.csv'.format(i), sep='\t')
power_spec.columns = ['k', 'Pk']


power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=np.linspace(power_spec['k'].iloc[1],power_spec['k'].iloc[-10], 200))).mean()
power_spec_grouped2 = power_spec.groupby(pd.cut(power_spec['k'], bins=np.logspace(-2,1.3, 300))).mean()
# pdb.set_trace()

fig2, ax2 = plt.subplots(dpi=120)
plot2 = ax2.plot(power_spec_grouped1['k'],power_spec_grouped1['Pk'])[0]
# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k ~~(h~\mathrm{Mpc}^{-1}$)")
ax2.set_ylabel(r"$P(k) ~~(h^{-1}\mathrm{Mpc})^3$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(top=1e5)
ax2.grid(True)

fig2.savefig(args.Pkdir+'/power_spec_{0:03d}.pdf'.format(0))


def update(i):
    power_spec = pd.read_csv(args.Pkdir+'Pk_{0:03d}.csv'.format(i), sep='\t')
    power_spec.columns = ['k', 'Pk']
    power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=np.linspace(power_spec['k'].iloc[1],power_spec['k'].iloc[-10], 200))).mean()
    plot2.set_xdata(power_spec_grouped1['k'])
    plot2.set_ydata(power_spec_grouped1['Pk'])
    ax2.set_title("Non-linear power spectrum from {}th snapshot.".format(i))


anim = matplotlib.animation.FuncAnimation(fig2, update, frames=201, interval=200)

anim.save(args.Pkdir+"/non-linear_power_spectrum.mp4")#, writer="imagemagick")
print("saved")


# 









