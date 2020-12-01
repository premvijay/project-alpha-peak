import numpy as np
import pandas as pd
import tables

import os
import sys
import pickle, json
from time import time, sleep

from gadget_tools import Snapshot, read_positions_all_files, read_velocities_all_files
from pm_tools import assign_density, project_to_slice, Region, Transform
from field_tools import compute_power_spec

import socket
# from mpi4py import MPI
import argparse
import pdb
import gc


parser = argparse.ArgumentParser(
    description='Assign density',
    usage= 'python')

parser.add_argument('--simdir', type=str, default='/scratch/aseem/sims', 
                help='Directory path for all simulations')
parser.add_argument('--simname', type=str, default='bdm_cdm1024',
                help='Simulation directory name')
parser.add_argument('--rundir', type=str, default='r1',
                help='Directory name containing the snapshot binaries')

parser.add_argument('--snap_i', type=int, default=200, help='Snapshot index number')

parser.add_argument('--downsample', type=int, default=1, 
                help='Downsample the particles in simulation by this many times')

parser.add_argument('--tree_root', type=int, default=200)
parser.add_argument('--M_around', type=float, default=3e12)
parser.add_argument('--max_halos', type=int, default=1000)

parser.add_argument('--halos_file_suffix', type=str, default='',
                help='halo file suffix like _1')

parser.add_argument('--scheme', type=str, default='TSC',
                help='Scheme for assigning particles to grid')

parser.add_argument('--grid_size', type=int, default=512,
                help='Grid size : number of cells along each direction')

parser.add_argument('--align', action='store_true', help='Save aligned and then stacked images')

parser.add_argument('--noalign', action='store_true', help='Save stacked images without any alignment')

parser.add_argument('--use_existing', type=int, default=0, help='Continue from existing stack')

parser.add_argument('--slice2D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--phase_space_1D', action='store_true', help='stack radial phase-space information of individual particles')

parser.add_argument('--phase_space_hist_1D', action='store_true', help='phase-space radial density')

parser.add_argument('--outdir', type=str, default='/scratch/cprem/sims',
                help='Directory to save the requested output')



args = parser.parse_args()

if not args.align and not args.noalign:
    args.noalign = True

# simdir = '/scratch/aseem/sims' if args.simdir is None else args.simdir
# simname = 'bdm_cdm1024' if args.simname is None else args.simname
# outdir = '/scratch/cprem/sims' if args.outdir is None else args.outdir

# scheme = 'CIC' if args.scheme is None else args.scheme
# grid_size = 512 if args.grid_size is None else args.grid_size

snapdir = os.path.join(args.simdir, args.simname, args.rundir)
halosfile = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', 'halos_list', f'halos_select_{args.M_around:.1e}_{args.max_halos:d}{args.halos_file_suffix:s}.csv')
halosfile = os.path.join('/scratch/cprem/sims', args.simname, args.rundir, 'halo_centric', 'halos_list', f'halos_select_{args.M_around:.1e}_{args.max_halos:d}{args.halos_file_suffix:s}.csv')

outdir = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', args.scheme, '{0:d}'.format(args.grid_size) )
os.makedirs(outdir, exist_ok=True)

print('Hostname is', socket.gethostname() )

t_now = time()



print('\n Starting to read snapshots binaries')

sys.stdout.flush()

filename_prefix = 'snapshot_{0:03d}'.format(args.snap_i)
filepath_prefix = os.path.join(snapdir, filename_prefix)

filepath = filepath_prefix + '.0'
print(filepath)
snap = Snapshot()
snap.from_binary(filepath)


posd = read_positions_all_files(filepath_prefix, downsample=args.downsample)

# posd = posd[:10000]


print('\n Particle positions read from all binaries in the snapshot')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

sys.stdout.flush()

veld = read_velocities_all_files(filepath_prefix, downsample=args.downsample)

veld /= (1+snap.redshift)**0.5

print('\n Particle velocities read from all binaries in the snapshot')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

sys.stdout.flush()


halos = pd.read_csv(halosfile, engine='c', index_col='id(1)')
halos_this_step = halos[halos['Snap_num(31)']==args.snap_i]
halos_root = halos[halos['Snap_num(31)']==args.tree_root]

# choose cube size
R_vir = halos_this_step['rvir(11)'].mean() / 1e3
R_vir_root = halos_root['rvir(11)'].mean() / 1e3
M_vir_root = halos_root['mvir(10)'].mean()

v_vir_root = 6.558138e-5 * (M_vir_root/R_vir_root)**(0.5)

L_cube = 10 * R_vir_root
R_sphere_focus = np.sqrt(3)/2 * L_cube * (1+1e-2)
L_cube_focus = np.sqrt(3) * L_cube * (1+1e-2)

slice_thickness = 4*R_vir_root

ps_r_max_vir = 8
ps_r_max = ps_r_max_vir * R_vir_root

ps_vr_max_vir = 3
ps_vr_max = ps_vr_max_vir * v_vir_root

mean_dens = posd.shape[0]/ (args.grid_size * snap.box_size / L_cube)**3

print('\n halos list read from the file')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

max_halos_total = args.use_existing + args.max_halos

phasedir = os.path.join(outdir,'phase-space')
metadir = os.path.join(outdir,'meta')
# if not args.align:
#     phasedir = os.path.join(phasedir, 'unaligned')
os.makedirs(phasedir, exist_ok=True)
os.makedirs(metadir, exist_ok=True)

slicedir_aligned = os.path.join(outdir,'slice2D')
slicedir_unaligned = os.path.join(slicedir_aligned, 'unaligned')
# if not args.align:
#     slicedir = os.path.join(slicedir, 'unaligned')
os.makedirs(slicedir_unaligned, exist_ok=True)
os.makedirs(slicedir_aligned, exist_ok=True)

if args.use_existing:
    with open(os.path.join(metadir, f'meta_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.json'), 'rt') as metafile:
        metadict = json.load(metafile)
    j = metadict['N_stack']

    print(f'\n Continuing from existing stack of {j:d}')
    t_bef, t_now = t_now, time()
    print(t_now-t_bef)

    if args.slice2D:
        if args.noalign:
            delta2D_unaligned = np.load( os.path.join(slicedir_unaligned, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.npy') )
        if args.align:
            delta2D_aligned = np.load( os.path.join(slicedir_aligned, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.npy') )
        
    if args.phase_space_hist_1D:
        rad_ps_hist = np.load( os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.npy') )

    if args.phase_space_1D:
        h5file_phase = tables.open_file(os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.hdf5'), mode='a')
        rad = h5file_phase.root.radius
        rad_vel = h5file_phase.root.radial_velocity
else:
    j = 0
    if args.slice2D:
        if args.noalign:
            delta2D_unaligned = np.zeros((args.grid_size,)*2, dtype=np.float64)
        if args.align:
            delta2D_aligned = np.zeros((args.grid_size,)*2, dtype=np.float64)
        # assert not np.isnan(delta2D).any(), 'before looping'
    if args.phase_space_hist_1D:
        rad_ps_hist = np.zeros((1024,)*2, dtype=np.float64)
    if args.phase_space_1D:
        h5file_phase = tables.open_file(os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.hdf5'), mode='w')
        atom = tables.Float64Atom()
        rad = h5file_phase.create_earray(h5file_phase.root, 'radius', atom, shape=(0,))
        rad_vel = h5file_phase.create_earray(h5file_phase.root, 'radial_velocity', atom, shape=(0,))

 

print('\n Starting with first halo')

for h in halos_this_step.index:
    t_now1 = time()

    halo_cen = halos_this_step[['x(17)','y(18)','z(19)']].loc[h].to_numpy()
    # region = Region('cube', cen=halo_cen,side=L_cube_focus,box_size=snap.box_size)
    region = Region('sphere', cen=halo_cen,rad=R_sphere_focus,box_size=snap.box_size)
    select_index = region.selectPrtcl(posd, engine='c++')

    print('\n particles selected in a covering region')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)

    posd_focus = region.shift_origin(posd[select_index])
    # posd_focus =

    print('\n particle positions shifted to origin at halo centre')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)


    if args.slice2D:
        if args.align:
            rot_vec = halos_this_step[['A[x](45)','A[y](46)','A[z](47)']].loc[h].to_numpy()
            if not np.isclose(rot_vec,0).all():
                posd_focus_aligned = Transform.rotate(posd_focus, rot_vec)
            else:
                print("\n Can\'t rotate for this halo, probably spherical")
                posd_focus_aligned = posd_focus.copy()
        if args.noalign:
            posd_focus_unaligned = posd_focus.copy()
                

        print('\n particle positions rotated')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

        # posd_focus += L_cube/2
        if args.noalign:
            posd_cube_unaligned = posd_focus_unaligned[np.all((posd_focus_unaligned>-L_cube/2) & (posd_focus_unaligned<L_cube/2), axis=1)]

        if args.align:
            posd_cube_aligned = posd_focus_aligned[np.all((posd_focus_aligned>-L_cube/2) & (posd_focus_aligned<L_cube/2), axis=1)]

        
        # posd_cube = posd_focus[np.all((posd_focus>-L_cube/2) & (posd_focus<L_cube/2), axis=1)]
        # posd_cube += L_cube/2

        print('\n particles selected for the grid')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)
        
        if args.noalign:
            particle_grid_unaligned = assign_density(posd_cube_unaligned+L_cube/2, L_cube, args.grid_size, scheme=args.scheme, overdensity=False)
        if args.align:
            particle_grid_aligned = assign_density(posd_cube_aligned+L_cube/2, L_cube, args.grid_size, scheme=args.scheme, overdensity=False)

        
        print('\n density assignment is done')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

        # assert not np.isnan(particle_grid_aligned).any(), 'dens assigned'

        # delta_j = (particle_grid_aligned / mean_dens) - 1
        if args.noalign:
            delta2D_j_unaligned = project_to_slice(particle_grid_unaligned, L_cube, axis=2, around_position='centre', thick=slice_thickness)
        if args.align:
            delta2D_j_aligned = project_to_slice(particle_grid_aligned, L_cube, axis=2, around_position='centre', thick=slice_thickness)

        print('\n 2d projected density is obtained')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

        # assert not np.isnan(delta2D_j).any(), 'delta'
        if args.noalign:
            delta2D_unaligned *= j
            delta2D_unaligned += delta2D_j_unaligned
            delta2D_unaligned /= j+1

            np.save(os.path.join(slicedir_unaligned, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), delta2D_unaligned)

        if args.align:
            delta2D_aligned *= j
            delta2D_aligned += delta2D_j_aligned
            delta2D_aligned /= j+1    

            np.save(os.path.join(slicedir_aligned, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), delta2D_aligned)
        
        print('\n delta is stacked')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

        # assert not np.isnan(delta2D).any(), 'delta'

    if args.phase_space_1D or args.phase_space_hist_1D:
        rad_j = np.linalg.norm(posd_focus, axis=1)
        rad_vel_j = (veld[select_index] * posd_focus).sum(axis=1) / rad_j
        print('\n radial velocities and positions computed')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

    if args.phase_space_1D:
        rad.append(rad_j)
        rad_vel.append(rad_vel_j)
        print('\n  individual radial velocities and positions stacked to hdf5')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

    if args.phase_space_hist_1D:
        rad_ps_hist *= j
        rad_ps_hist += np.histogram2d(rad_j, rad_vel_j, bins=[np.linspace(0,ps_r_max,1025),np.linspace(-ps_vr_max,ps_vr_max,1025)], density=True)[0]*rad_j.shape[0]
        rad_ps_hist /= j+1
        np.save(os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), rad_ps_hist)
        print('\n  radial phase space distribution stacked as hist')
        t_bef1, t_now1 = t_now1, time()
        print(t_now1-t_bef1)

    j+=1

    with open(os.path.join(metadir, f'meta_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.json'), 'w') as metafile:
        dict = {'N_stack':j, 'L_cube':L_cube, 'R_vir':R_vir, 'R_vir_root':R_vir_root, 'M_vir_root':M_vir_root, 'v_vir_root':v_vir_root, 
        'slice_thickness':slice_thickness, 'ps_r_max':ps_r_max, 'ps_r_max_vir':ps_r_max_vir, 'ps_vr_max':ps_vr_max, 'ps_vr_max_vir':ps_vr_max_vir, }
        json.dump(dict,metafile, indent=True)
        # file.write(i)

    print('\n {0} number of halo-centric images stacked at snapshot {1}'.format(j, args.snap_i))
    t_bef, t_now = t_now, time()
    print('total time per halo', t_now-t_bef)
    
    sys.stdout.flush()
    # pdb.set_trace()



del posd, veld
gc.collect()

if args.slice2D:
    if args.noalign:
        np.save(os.path.join(slicedir_unaligned, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), delta2D_unaligned)
    if args.align:
        np.save(os.path.join(slicedir_aligned, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), delta2D_aligned)

if args.phase_space_hist_1D:
    np.save(os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), rad_ps_hist)

if args.phase_space_1D:
    h5file_phase.close()

with open(os.path.join(metadir, f'meta_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.json'), 'w') as metafile:
    dict = {'N_stack':j, 'L_cube':L_cube, 'R_vir':R_vir, 'R_vir_root':R_vir_root, 'slice_thickness':slice_thickness, 'ps_r_max':ps_r_max, 'ps_r_max_vir':ps_r_max_vir, 'ps_vr_max':ps_vr_max}
    json.dump(dict,metafile, indent=True)


print('\n density assigned to grid around halos for snapshot {0:03d}'.format(args.snap_i))



# delta = particle_grid 

# if args.slice2D:
#     slicedir = os.path.join(outdir,'slice2D')
#     os.makedirs(slicedir, exist_ok=True)
#     np.save(os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}.npy'.format(args.snap_i, args.downsample) ), delta2D)



# pdb.set_trace()











