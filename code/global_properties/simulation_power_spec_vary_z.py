import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation
import pandas as pd
import argparse
import pdb
from munch import Munch

import camb
from camb import model, initialpower

from gadget_tools import Snapshot
from fitting_fns import halofit


parser = argparse.ArgumentParser(
    description='Density field evolution from Gadget simulation.',
    usage= 'python ./visualize_simulation.py')

parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--cosmo', type=str, default='P18', help='cosmology parameters data from')
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
p = schemes.index(scheme) + 1

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', f'full_box')
os.makedirs(plotsdir, exist_ok=True)

simdir = os.path.join('/scratch/aseem/sims', args.simname, args.rundir)


# cosmology = 'P18' if args.simname=='bdm_cdm1024' else 'WMAP7'
if args.cosmo =='P18':
    cos_par_vals = (0.306, 0.694, 0.0484, 0.678, 0.9677, 0.815)
elif args.cosmo=='WMAP7':
    cos_par_vals = (0.276, 0.724, 0.045, 0.7, 0.961, 0.811)

cos_pars = Munch()
cos_pars.Om0, cos_pars.Ode0, cos_pars.Ob0, cos_pars.h, cos_pars.ns, cos_pars.sig8 = cos_par_vals
cos_pars.Ombh2 = (cos_pars.Ob0)*cos_pars.h**2
cos_pars.Omch2 = (cos_pars.Om0-cos_pars.Ob0)*cos_pars.h**2

print(args.cosmo, vars(cos_pars))

def Omega(z, Om0):
    E = Om0 * (1+z)**3 + (1-Om0)
    return Om0 * (1+z)**3 / E

def D1(z, Om0):
    Om_m = Omega(z, Om0)
    Om_L = 1 - Om_m
    return 5/2* 1/(1+z) * Om_m / (Om_m**(4/7) - Om_L + (1+Om_m/2)*(1+Om_L/70))

def adjust_lightness(color, amount=0.5):
    # This function from stack for creating lighter or darker versions of matplotlib color
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def lighter(color): return adjust_lightness(color, 1.5)

def darker(color): return adjust_lightness(color, 0.7)

# simnames = ['bdm_cdm1024'

# transfer_func_file = '/mnt/home/faculty/caseem/config/transfer/classTf_om0.14086_Ok0.0_ob0.02226_h0.6781_ns0.9677.txt'

# transfer_df = pd.read_csv(transfer_func_file, sep='\s+',header=None)

i_list = list(range(50,201,50))

snap = Snapshot(os.path.join(simdir, f'snapshot_{200:03d}.0'))

box_size = snap.box_size
k_nyq = np.pi * grid_size / snap.box_size
k_start = 2* np.pi / snap.box_size

redshifts=[]
for i in i_list:
    snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))
    redshifts.append(snap.redshift)


#Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars_camb = camb.CAMBparams()
pars_camb.set_cosmology(H0=cos_pars.h*100, ombh2=cos_pars.Ombh2, omch2=cos_pars.Omch2)
pars_camb.InitPower.set_params(ns=cos_pars.ns)
#Note non-linear corrections couples to smaller scales than you want
pars_camb.set_matter_power(redshifts=redshifts, kmax=10*k_nyq)

#Linear spectra
pars_camb.NonLinear = model.NonLinear_none
results_camb = camb.get_results(pars_camb)
kh_camb, z_camb, pk_camb = results_camb.get_matter_power_spectrum(minkh=2e-7, maxkh=100, npoints = 5000)
# s8 = np.array(results.get_sigma8())

#Non-Linear spectra (Halofit)
pars_camb.NonLinear = model.NonLinear_both
results_camb.calc_power_spectra(pars_camb)
kh_camb_nonlin, z_camb_nonlin, pk_camb_nonlin = results_camb.get_matter_power_spectrum(minkh=k_start*0.8, maxkh=1.2*k_nyq, npoints = 200)






fig1, ax2 = plt.subplots(1, figsize=(7.5,7), dpi=150)
plt.rcParams['lines.linewidth'] = 1

# ax2.plot([],[], ' ', label=f"Scheme-{scheme}, Grid-size: {grid_size:d}")



for index, i in enumerate(i_list[::-1]):
    snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

    savesdir = os.path.join('/scratch/cprem/sims', args.simname, args.rundir, 'global', scheme, '{0:d}'.format(grid_size))
    print(savesdir)

    color=next(ax2._get_lines.prop_cycler)['color']
    # lightcolor = adjust_lightness(color, 0.5)
    # darkcolor = adjust_lightness(color, 2)

    # transfer_df = pd.read_csv('/mnt/home/faculty/caseem/config/transfer/classTf_om0.14086_Ok0.0_ob0.02226_h0.6781_ns0.9677.txt', sep='\s+',header=None)

    # k_full = transfer_df[0]
    # pk_lin = transfer_df[1]**2*transfer_df[0] * (D1(snap.redshift, snap.Omega_m_0)/ D1(0, snap.Omega_m_0))**2

    k_full = kh_camb
    pk_lin = pk_camb[index]


    ax2.plot(k_full, pk_lin, color=color, linestyle=(0, (1, 1)), zorder=3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    # ax2.plot(transfer_df[0], transfer_df[1]**2*transfer_df[0]/ (1+snap.redshift), color=color, linestyle='dotted', label='linear theory')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')

    ax2.plot(kh_camb_nonlin, pk_camb_nonlin[index], linestyle='solid', color=lighter(color), zorder=1)

    pk_fit = halofit.NonLinPowerSpecCDM(Omega(snap.redshift, snap.Omega_m_0))
    pk_fit.set_Del2L_interpolate(k_full, pk_lin)
    pk_fit.compute_params()
    print(vars(pk_fit))
    ax2.plot(kh_camb_nonlin, pk_fit.P(kh_camb_nonlin), linestyle='solid', color=darker(color), zorder=2)

    Pkdir = os.path.join(savesdir,'power_spectrum')
    # infodir = os.path.join(savesdir,'info')
    
    power_spec = pd.read_csv(os.path.join(Pkdir, 'Pk_{0:03d}.csv'.format(i)), sep='\t', dtype='float64')
    power_spec.columns = ['k', 'Pk']

    if interlaced:
        power_spec_inlcd = pd.read_csv(os.path.join(Pkdir, 'Pk{1}_{0:03d}.csv'.format(i,inlcd_str)), sep='\t', dtype='float64')
        power_spec_inlcd.columns = ['k', 'Pk']

    lin_bin = np.linspace(np.sqrt(k_start*k_nyq),k_nyq, 30)
    log_bin = np.logspace(np.log10(k_start),np.log10(k_nyq), 25)
    # merge_bin = np.concatenate([lin_bin,log_bin])
    # merge_bin.sort()power_spec['k'].iloc[1]
    merge_bin = np.union1d(lin_bin, log_bin)


    # power_spec_grouped1 = power_spec.groupby(pd.cut(power_spec['k'], bins=lin_bin)).mean()
    # ax2.plot(power_spec_grouped1['k'],power_spec_grouped1['Pk'], color=color, linestyle='dashed', label=f"{scheme:s} scheme without interlacing")[0]

    if interlaced:
        power_spec_inlcd_grouped1 = power_spec_inlcd.groupby(pd.cut(power_spec_inlcd['k'], bins=merge_bin)).mean()
        power_spec_inlcd_grouped1['W_correct'] = np.sinc(power_spec_inlcd_grouped1['k']/(2*k_nyq))**(2*p+1)
        ax2.plot(power_spec_inlcd_grouped1['k'],power_spec_inlcd_grouped1['Pk']/ power_spec_inlcd_grouped1['W_correct'], color=color, linestyle='dashed', label=f"z={f'{snap.redshift:.3f}'.rstrip('0').rstrip('.'):s} in snapshot-{i:03d}")[0]
        ax2.scatter(power_spec_inlcd_grouped1['k'],power_spec_inlcd_grouped1['Pk']/ power_spec_inlcd_grouped1['W_correct'], color=color, s=4)

    power_spec_existing = pd.read_csv(os.path.join(simdir,f"Pk_{i:03d}.txt"),comment='#', sep='\t',names=['k','pk','ph','pcross'])
    power_spec_existing.plot('k','pk', loglog=True, ax=ax2, color=lighter(color), linestyle='dashed', label='', legend=False)

ax2.plot([],[], ' ', label=f"From GADGET simulation")
ax2.plot([],[], linestyle='dashed', color='gray', label=f"  our code {scheme}-{grid_size:d}")
ax2.plot([],[], linestyle='dashed', color=lighter('gray'), label=f"  for reference")
ax2.plot([],[], ' ', label=f"Halofit model")
ax2.plot([],[], linestyle='solid', color=darker('gray'), label='  Takahashi, et al. 2012')
ax2.plot([],[], linestyle='solid', color=lighter('gray'), label='  CAMB non-linear')
ax2.plot([],[], ' ', label=f"linear theory")
ax2.plot([],[], linestyle=(0, (1, 1)), color='gray', label='  CAMB linear')

# pdb.set_trace()


# ax2[1].plot(power_spec['lam'],power_spec['Pk'])
ax2.set_xlabel(r"$k$  ($h$ Mpc$^{-1}$)")
ax2.set_ylabel(r"$P(k)$  ($h^{-1}$Mpc)$^3$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(k_start*0.8,1.2*k_nyq)
ax2.set_ylim(top=1e5, bottom=2e-1)
ax2.grid(True)
fig1.suptitle(f"Matter power spectrum from {args.simname:s} simulation at different redshifts")
ax2.legend()

plt.tight_layout()

fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_vary_z_{scheme}{theme:s}.pdf'))
fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_vary_z_{scheme}{theme:s}.png'))
# fig1.savefig(os.path.join(plotsdir, f'single_snapshot_pk_{i:03d}{theme:s}.svg'))