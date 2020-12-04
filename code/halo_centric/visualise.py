import os
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation
from matplotlib import patches as mpatches
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import argparse
import pdb

from gadget_tools import Snapshot



parser = argparse.ArgumentParser(
    description='Density field evolution from Gadget simulation.',
    usage= 'python ./visualize_simulation.py')

parser.add_argument('--simname', type=str, default='bdm_cdm1024', help='Directory name containing the saved data')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--downsample', type=int, default=1, 
                help='visualize the Downsampled particles in simulation by this many times')

parser.add_argument('--tree_root', type=int, default=200)
parser.add_argument('--M_around', type=float, default=3e12)
parser.add_argument('--max_halos', type=int, default=1000)

parser.add_argument('--align', type=int, default=1, help='Visualize aligned and then stacked images')

parser.add_argument('--outdir', type=str, default='/scratch/cprem/sims')
parser.add_argument('--plots_into', type=str, default='/mnt/home/student/cprem/project-alpha-peak/notes_and_results/plots_and_anims')

parser.add_argument('--phase_space_hist_1D', action='store_true', help='phase-space radial density')

parser.add_argument('--snap_i', type=int, default=200, help='Snapshot index number')

parser.add_argument('--light_snaps', type=int, default=0, help='save white bg images for pdf notes')


args = parser.parse_args()

grid_size = 512
scheme = 'TSC'

save_delta = False
save_phasespace = False
save_delta = True
# save_phasespace = True


# dark_all = True
if args.light_snaps:
    theme = '_light'
    plt.style.use('default')
    plt.set_cmap('nipy_spectral')
else:
    theme = ''
    plt.style.use('dark_background')
    plt.set_cmap('nipy_spectral')


# simname = 'bdm_cdm1024' if args.simname is None else args.simname

savesdir_global = os.path.join(args.outdir, args.simname, args.rundir, 'TSC', '512')

savesdir = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', scheme, f'{grid_size:d}')

print(savesdir)

halosfile = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', 'halos_list', f'halos_select_{args.M_around:.1e}_{args.max_halos:d}.csv')

simdir = os.path.join('/scratch/aseem/sims', args.simname, args.rundir)
slicedir = os.path.join(savesdir,'slice2D')
phasedir = os.path.join(savesdir,'phase-space')
metadir = os.path.join(savesdir,'meta')
infodir = os.path.join(savesdir_global,'info')

plotsdir = os.path.join(args.plots_into, f'{args.simname:s}_{args.rundir:s}', f'halo_centric_{scheme:s}_{grid_size:d}')
os.makedirs(plotsdir, exist_ok=True)

align_str = ''
if not args.align:
    slicedir = os.path.join(slicedir, 'unaligned')
    align_str += '_unaligned'

def Omega(z, Om0):
    E = Om0 * (1+z)**3 + (1-Om0)
    return Om0 * (1+z)**3 / E

def Del_vir(Om):
    x = Om - 1
    return (18*np.pi**2 + 82*x - 39*x**2)/Om

halos = pd.read_csv(halosfile, engine='c', index_col='id(1)')
halos_root = halos[halos['Snap_num(31)']==args.tree_root]

i = args.snap_i

halos_this_step = halos[halos['Snap_num(31)']==i]


# with open(os.path.join(infodir, f'header_{i:03d}.p'), 'rb') as infofile:
#     snap=pickle.load(infofile)

snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

mean_dens_comoving = np.dot(snap.mass_table*1e10, snap.N_prtcl_total) / snap.box_size**3

with open( os.path.join(metadir, f'meta_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.json'), 'rt' ) as metafile:
    metadict = json.load(metafile)

box_size = metadict['L_cube']
slice_thickness = metadict['slice_thickness']

M_vir_median = halos_root['mvir(10)'].median()
M_vir_range = ( halos_root['mvir(10)'].min(), halos_root['mvir(10)'].max() )


# Omega = lambda z : Omega(z, snap.Omega_m_0)


delta_slice = np.load( os.path.join(slicedir, f'slice_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.npy') )

phase_space_1D = np.load( os.path.join(phasedir, f'phase-space_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.npy') )


fig1, ax1 = plt.subplots(figsize=(9,7.5))#, dpi=120)

mass_unit = r'$h^{-1}M_{\odot}$'

fig1.suptitle(f"Simulation: {args.simname}, Grid size: {grid_size}, Scheme: {scheme};     Snapshot-{i:03d} at redshift z={snap.redshift:.4f};\n Halos selected by mass at redshift 0 in [{M_vir_range[0]:.2e},{M_vir_range[1]:.2e}] {mass_unit:s} with median {M_vir_median:.2e} {mass_unit:s}")

im1 = ax1.imshow(delta_slice+1+1e-5, extent=[-box_size/2,box_size/2,-box_size/2,box_size/2], norm=LogNorm(vmin=3e-1))
cb1 = fig1.colorbar(im1,ax=ax1)
cb1.set_label(r"$(1+\delta)$")
# ax1.set_title(f"{slice_thickness:.3f} {r'$h^{-1}$':s} Mpc thick slice in halo centric stack of {metadict['N_stack']}")
ax1.set_title(r'$4 ~R_{\rm{vir}}(z=0)$' + f" = {slice_thickness:.3f} {r'$h^{-1}$':s} Mpc thick slice in halo centric stack of {metadict['N_stack']}")
ax1.set_xlabel(r"$h^{-1}$Mpc")
ax1.set_ylabel(r"$h^{-1}$Mpc")

# these are matplotlib.patch.Patch properties
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
# ax1.text(0.05, 0.5, r'$4 ~R_{\rm{vir}}(z=0)$' + f" = {metadict['R_vir_root']:.3f} {r'$h^{-1}$':s} Mpc", transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

at = AnchoredText(r'$R_{\rm{vir}}(z=0)$' + f" = {metadict['R_vir_root']:.3f} {r'$h^{-1}$':s} Mpc", loc='upper left', prop=dict(backgroundcolor='purple', alpha=1, color='yellow',size=12), frameon=True)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)


# Make circle of virial radius from catalogue
r_vir_circ = mpatches.Circle((0,0),radius=metadict['R_vir'], fill=False, label='virial radius from halo catalogue')
ax1.add_patch(r_vir_circ)

if args.align:
    axisratio_2_1 = (halos_this_step['b_to_a(43)'].mean() * halos_this_step['c_to_a(44)'].mean() )**(1/2)
    #(halos_this_step['b_to_a(43)'].mean()**2 + halos_this_step['c_to_a(44)'].mean()**2 )**(1/2)
    r_vir_ell = mpatches.Ellipse((0,0), height=metadict['R_vir']*2, width=metadict['R_vir']*2*axisratio_2_1, fill=False, label='virial boandary based on halo catalogue')
    ax1.add_patch(r_vir_ell)

# M_vir = args.M_around 
M_vir = halos_this_step['mvir(10)'].mean()
R_vir_sc = ( M_vir / (4/3 * np.pi * Del_vir(Omega(snap.redshift, snap.Omega_m_0)) * mean_dens_comoving) )**(1/3)
r_vir_sc_circ = plt.Circle((0,0),radius=R_vir_sc, fill=False, ls='--', label='virial radius from spherical collapse')
ax1.add_patch(r_vir_sc_circ)

plt.legend()
# ax1.set_xscale('log')


plt.tight_layout()

if save_delta:
    fig1.savefig(os.path.join(plotsdir, f'single_snapshot{align_str}_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}{theme:s}.pdf'))
    fig1.savefig(os.path.join(plotsdir, f'single_snapshot{align_str}_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}{theme:s}.png'))
    fig1.savefig(os.path.join(plotsdir, f'single_snapshot{align_str}_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}{theme:s}.svg'))

def update(i):
    print(i, 'starting')
    delta_slice = np.load( os.path.join(slicedir, f'slice_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.npy') )
    im1.set_data(delta_slice+1+1e-5)
    
    with open( os.path.join(metadir, f'meta_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.json'), 'rt' ) as metafile:
        metadict = json.load(metafile)
    # with open(os.path.join(infodir, f'header_{i:03d}.p'), 'rb') as infofile:
    #     snap=pickle.load(infofile)
    snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

    halos_this_step = halos[halos['Snap_num(31)']==i]
    M_vir = halos_this_step['mvir(10)'].mean()
    R_vir_sc = ( M_vir / (4/3 * np.pi * Del_vir(Omega(snap.redshift, snap.Omega_m_0)) * mean_dens_comoving) )**(1/3)

    r_vir_circ.set_radius(metadict['R_vir'])

    if args.align:
        axisratio_2_1 = (halos_this_step['b_to_a(43)'].mean() * halos_this_step['c_to_a(44)'].mean() )**(1/2)
        r_vir_ell.set_height(metadict['R_vir']*2)
        r_vir_ell.set_width(metadict['R_vir']*2*axisratio_2_1)
        
    r_vir_sc_circ.set_radius(R_vir_sc)

    ax1.set_title(r'$4 ~R_{\rm{vir}}(z=0)$' + f" = {slice_thickness:.3f} {r'$h^{-1}$':s} Mpc thick slice in halo centric stack of {metadict['N_stack']}")

    
    fig1.suptitle(f"Simulation: {args.simname}, Grid size: {grid_size}, Scheme: {scheme};     Snapshot-{i:03d} at redshift z={snap.redshift:.4f};\n Halos selected by mass at redshift 0 in [{M_vir_range[0]:.2e},{M_vir_range[1]:.2e}] {mass_unit:s} with median {M_vir_median:.2e} {mass_unit:s}")


anim = matplotlib.animation.FuncAnimation(fig1, update, frames=range(6,args.tree_root+1), interval=500)

# plt.rcParams['animation.ffmpeg_path'] = ''

# Writer=matplotlib.animation.ImageMagickWriter
Writer=matplotlib.animation.FFMpegWriter
writer = Writer(fps=10)

if not args.light_snaps and save_delta:
    anim.save(os.path.join(plotsdir, f'simulation_visualisation{align_str}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.mp4'), writer=writer, dpi=100)
    print("saved video density delta")

i = args.snap_i

with open( os.path.join(metadir, f'meta_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.json'), 'rt' ) as metafile:
    metadict = json.load(metafile)
# with open(os.path.join(infodir, f'header_{i:03d}.p'), 'rb') as infofile:
#     snap=pickle.load(infofile)
snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))


fig1, ax1 = plt.subplots(figsize=(12,7.5))
im1 = ax1.imshow(phase_space_1D.T*snap.mass_table[1]*1e7, norm=LogNorm(), extent=[0,metadict['ps_r_max_vir'],-metadict['ps_vr_max_vir'],metadict['ps_vr_max_vir']], aspect='auto')
# plt.xlim(0,10)
cb1 = fig1.colorbar(im1, ax=ax1)
ax1.set_xlim(0,3)
# ax1.set_xlabel(r'Radius in units of $~R_{\rm{vir}}(z=0)$' + f" = {metadict['R_vir_root']:.3f} {r'$h^{-1}$':s} Mpc")
# ax1.set_ylabel(r'Peculiar velocity in units of $~v_{\rm{vir}}(z=0)$' + f" = {metadict['v_vir_root']:.3f} km/s")
# cb1.set_label(mass_unit + r' / (km~s$^{-1}$) / ($h^{-1}$ Mpc)')
cb1.set_label(r'$M_{\odot}$ / (km/s) / (kpc)')
ax1.set_xlabel(r'Distance from halo centre in units of $~R_{\rm{vir}}(z=0)$' + f" = {metadict['R_vir_root']:.3f} {r'$h^{-1}$':s} Mpc")
ax1.set_ylabel(r'Peculiar velocity radially away from halo in units of $~v_{\rm{vir}}(z=0)$' + f" = {metadict['v_vir_root']:.3f} km/s")
ax1.set_title(f"Radial phase space density - averaged over {metadict['N_stack']} halos")

fig1.suptitle(f"Simulation: {args.simname}, Grid size: {grid_size}, Scheme: {scheme};     Snapshot-{i:03d} at redshift z={snap.redshift:.4f};\n Halos selected by mass at redshift 0 in [{M_vir_range[0]:.2e},{M_vir_range[1]:.2e}] {mass_unit:s} with median {M_vir_median:.2e} {mass_unit:s}")


plt.tight_layout()

if save_phasespace:
    fig1.savefig(os.path.join(plotsdir, f'phase_space_1D_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}{theme:s}.pdf'))
    fig1.savefig(os.path.join(plotsdir, f'phase_space_1D_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}{theme:s}.png'))
    fig1.savefig(os.path.join(plotsdir, f'phase_space_1D_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}{theme:s}.svg'))

def update_phase_space(i):
    print(i, 'starting')
    phase_space_1D = np.load( os.path.join(phasedir, f'phase-space_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.npy') )
    snap = Snapshot(os.path.join(simdir, f'snapshot_{i:03d}.0'))

    im1.set_data(phase_space_1D.T*snap.mass_table[1]*1e7)
    
    with open( os.path.join(metadir, f'meta_{i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.json'), 'rt' ) as metafile:
        metadict = json.load(metafile)
    # with open(os.path.join(infodir, f'header_{i:03d}.p'), 'rb') as infofile:
    #     snap=pickle.load(infofile)

    

    # halos_this_step = halos[halos['Snap_num(31)']==i]
    # M_vir = halos_this_step['mvir(10)'].mean()
    # R_vir_sc = ( M_vir / (4/3 * np.pi * Del_vir(Omega(snap.redshift, snap.Omega_m_0)) * mean_dens_comoving) )**(1/3)

    ax1.set_title(f"Radial phase space density - averaged over {metadict['N_stack']} halos")    

    fig1.suptitle(f"Simulation: {args.simname}, Grid size: {grid_size}, Scheme: {scheme};     Snapshot-{i:03d} at redshift z={snap.redshift:.4f};\n Halos selected by mass at redshift 0 in [{M_vir_range[0]:.2e},{M_vir_range[1]:.2e}] {mass_unit:s} with median {M_vir_median:.2e} {mass_unit:s}")



anim = matplotlib.animation.FuncAnimation(fig1, update_phase_space, frames=range(6,args.tree_root+1), interval=500)

# plt.rcParams['animation.ffmpeg_path'] = ''

# Writer=matplotlib.animation.ImageMagickWriter
Writer=matplotlib.animation.FFMpegWriter
writer = Writer(fps=10)

if not args.light_snaps and save_phasespace:
    anim.save(os.path.join(plotsdir, f'phase_space_1D_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.mp4'), writer=writer, dpi=100)
    print("saved video phase-space")


