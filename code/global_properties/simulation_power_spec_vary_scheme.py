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
from fitting_fns import halofit

# import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = .5

# plt.rcParams["lines.linewidth"] = 1

parser = argparse.ArgumentParser(
    description='Density field evolution from Gadget simulation.',
    usage= 'python ./visualize_simulation.py')

parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--plots_into', type=str, default='/mnt/home/student/cprem/project-alpha-peak/notes_and_results/plots_and_anims')

parser.add_argument('--snap_i', type=int, default=200, help='Snapshot index number')

parser.add_argument('--light_snaps', type=int, default=1, help='save white bg images for pdf notes')

args = parser.parse_args()

grid_size = 512
scheme = 'TSC'
# rundir = 'r1'
interlaced = True

inlcd_str = '_interlaced' if interlaced==True else ''

if args.light_snaps:
    theme = '_light'
    plt.style.use('default')
    # plt.set_cmap('nipy_spectral')
else:
    theme = ''
    plt.style.use('dark_background')
    # plt.set_cmap('nipy_spectral')

# simname = 'bdm_cdm1024' if args.simname is None else args.simname

schemes = ['NGP', 'CIC', 'TSC']

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', f'full_box')
os.makedirs(plotsdir, exist_ok=True)

i = args.snap_i

simdir = os.path.join('/scratch/aseem/sims', args.simname, args.rundir)
snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

box_size = snap.box_size

k_nyq = 2*np.pi * grid_size / snap.box_size

fig1, ax2 = plt.subplots(1, figsize=(7.5,7), dpi=150)

transfer_func_file = '/mnt/home/faculty/caseem/config/transfer/classTf_om0.14086_Ok0.0_ob0.02226_h0.6781_ns0.9677.txt'

transfer_df = pd.read_csv(transfer_func_file, sep='\s+',header=None)

def Omega(z, Om0):
    E = Om0 * (1+z)**3 + (1-Om0)
    return Om0 * (1+z)**3 / E

def D1(z, Om0):
    Om_m = Omega(z, Om0)
    Om_L = 1 - Om_m
    return 5/2* 1/(1+z) * Om_m / (Om_m**(4/7) - Om_L + (1+Om_m/2)*(1+Om_L/70))

k_full = transfer_df[0]
pk_lin = transfer_df[1]**2*transfer_df[0] * (D1(snap.redshift, snap.Omega_m_0)/ D1(0, snap.Omega_m_0))**2

ax2.plot(k_full, pk_lin, linestyle='dotted', label='linear theory')
ax2.set_xscale('log')
ax2.set_yscale('log')

# try:
#     power_spec_existing = pd.read_csv(os.path.join(simdir,f"Pk_{i:03d}.txt"),comment='#', sep='\t',names=['k','pk','ph','pcross'])
#     power_spec_existing.plot('k','pk', loglog=True, ax=ax2, linestyle='dashdot', label='reference non-linear')
# except:
#     print('no reference available')

pk_fit = halofit.NonLinPowerSpecCDM(Omega(snap.redshift, snap.Omega_m_0))
pk_fit.set_Del2L_interpolate(k_full, pk_lin)
pk_fit.compute_params()
print(vars(pk_fit))
ax2.plot(k_full, pk_fit.P(k_full), label='Halofit model')


ax2.plot([],[], ' ', label=f"Snapshot-{i:03d}, Grid-size: {grid_size:d}")



# fig1.suptitle("Simulation: {2}, ".format(i,snap.redshift,args.simname,grid_size)    )

for p,scheme in enumerate(schemes, start=1):
    savesdir = os.path.join('/scratch/cprem/sims', args.simname, args.rundir, 'global', scheme, '{0:d}'.format(grid_size))
    print(savesdir)

    Pkdir = os.path.join(savesdir,'power_spectrum')
    # infodir = os.path.join(savesdir,'info')

    # ax1.set_xscale('log')
    # p = schemes.index(scheme)
    
    
    power_spec = pd.read_csv(os.path.join(Pkdir, 'Pk_{0:03d}.csv'.format(i)), sep='\t', dtype='float64')
    power_spec.columns = ['k', 'Pk']

    if interlaced:
        power_spec_inlcd = pd.read_csv(os.path.join(Pkdir, 'Pk{1}_{0:03d}.csv'.format(i,inlcd_str)), sep='\t', dtype='float64')
        power_spec_inlcd.columns = ['k', 'Pk']

    lin_bin = np.linspace(power_spec['k'].iloc[1],8e0, 200)
    log_bin = np.logspace(-2,1.3, 100)
    merge_bin = np.concatenate([lin_bin,log_bin])
    merge_bin.sort()

    color=next(ax2._get_lines.prop_cycler)['color']

    power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=lin_bin)).mean()

    power_spec_grouped1['W_correct'] = np.sinc(power_spec_grouped1['k']/k_nyq)**(2*p)
    ax2.plot(power_spec_grouped1['k'], power_spec_grouped1['Pk']/ power_spec_grouped1['W_correct'], color=color, linestyle='solid', linewidth=0.5, label=f"{scheme:s} scheme without interlacing")[0]

    if interlaced:
        # power_spec_inlcd['W_correct'].iloc[::2] = 0
        power_spec_inlcd_grouped1 = power_spec_inlcd.groupby(pd.cut(power_spec_inlcd['k'], bins=lin_bin)).mean()

        power_spec_inlcd_grouped1['W_correct'] = np.sinc(power_spec_inlcd_grouped1['k']/k_nyq)**(2*p+1)
        ax2.plot(power_spec_inlcd_grouped1['k'],power_spec_inlcd_grouped1['Pk']/ power_spec_inlcd_grouped1['W_correct'], color=color, linestyle='dashed', linewidth=0.5, label=f"{scheme:s} scheme with interlacing")[0]



# pdb.set_trace()


# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k$  ($h$ Mpc$^{-1}$)")
ax2.set_ylabel(r"$P(k)$  ($h^{-1}$Mpc)$^3$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(5e-2,2e1)
ax2.set_ylim(top=1e5, bottom=2e-1)
ax2.grid(True)
ax2.set_title(f"Matter power spectrum from {args.simname:s} simulation at redshift z={f'{snap.redshift:.3f}'.rstrip('0').rstrip('.'):s} ")
ax2.legend()

plt.tight_layout()

fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_vary_scheme_{i:03d}{theme:s}.pdf'), dpi=600)
fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_vary_scheme_{i:03d}{theme:s}.png'), dpi=600)
# fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_{i:03d}{theme:s}.svg'))