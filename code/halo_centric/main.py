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

parser.add_argument('--snap_i', type=int, default=100, help='Snapshot index number')

parser.add_argument('--downsample', type=int, default=1, 
                help='Downsample the particles in simulation by this many times')

parser.add_argument('--tree_root', type=int, default=200)
parser.add_argument('--M_around', type=float, default=3e12)
parser.add_argument('--max_halos', type=int, default=500)

parser.add_argument('--halos_file_suffix', type=str, default='',
                help='halo file suffix like _1')

parser.add_argument('--scheme', type=str, default='CIC',
                help='Scheme for assigning particles to grid')

parser.add_argument('--grid_size', type=int, default=512,
                help='Grid size : number of cells along each direction')

parser.add_argument('--align', type=int, default=1, help='Align and then stack images')

parser.add_argument('--use_existing', type=int, default=0, help='Continue from existing stack')

parser.add_argument('--slice2D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--phase_space_1D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--outdir', type=str, default='/scratch/cprem/sims',
                help='Directory to save the requested output')



args = parser.parse_args()

# simdir = '/scratch/aseem/sims' if args.simdir is None else args.simdir
# simname = 'bdm_cdm1024' if args.simname is None else args.simname
# outdir = '/scratch/cprem/sims' if args.outdir is None else args.outdir

# scheme = 'CIC' if args.scheme is None else args.scheme
# grid_size = 512 if args.grid_size is None else args.grid_size

snapdir = os.path.join(args.simdir, args.simname, args.rundir)
halosfile = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', 'halos_list', f'halos_select_{args.M_around:.1e}_{args.max_halos:d}{args.halos_file_suffix:s}.csv')

outdir = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', args.scheme, '{0:d}'.format(args.grid_size) )
os.makedirs(outdir, exist_ok=True)

print('Hostname is', socket.gethostname() )

t_now = time()
print('\n Starting to read snapshots binaries')

filename_prefix = 'snapshot_{0:03d}'.format(args.snap_i)
filepath_prefix = os.path.join(snapdir, filename_prefix)

posd = read_positions_all_files(filepath_prefix, downsample=args.downsample)

# posd = posd[:10000]

veld = read_velocities_all_files(filepath_prefix, downsample=args.downsample)


print('\n Particle positions read from all binaries in the snapshot')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

filepath = filepath_prefix + '.0'
print(filepath)
snap = Snapshot()
snap.from_binary(filepath)



halos = pd.read_csv(halosfile, engine='c', index_col='id(1)')
halos_this_step = halos[halos['Snap_num(31)']==args.snap_i]
halos_root = halos[halos['Snap_num(31)']==args.tree_root]

# choose cube size
R_vir = halos_this_step['rvir(11)'].mean() / 1e3
R_vir_root = halos_root['rvir(11)'].mean() / 1e3
L_cube = 10 * R_vir_root
R_sphere_focus = np.sqrt(3)/2 * L_cube * (1+1e-2)
L_cube_focus = np.sqrt(3) * L_cube * (1+1e-2)

slice_thickness = 4*R_vir_root

mean_dens = posd.shape[0]/ (args.grid_size * snap.box_size / L_cube)**3

print('\n halos list read from the file')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

max_halos_total = args.use_existing + args.max_halos

phasedir = os.path.join(outdir,'phase-space')
if not args.align:
    phasedir = os.path.join(phasedir, 'unaligned')
os.makedirs(phasedir, exist_ok=True)

if args.use_existing:
    with open(os.path.join(slicedir, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.meta'), 'rt') as metafile:
        metadict = json.load(metafile)
    j = metadict['N_stack']
    if args.slice2D:
        delta2D = np.load( os.path.join(slicedir, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.use_existing:d}.npy') )
        print(f'\n Continuing from existing stack of {j:d}')
        t_bef, t_now = t_now, time()
        print(t_now-t_bef)
    if args.phase_space_1D:
        h5file_phase = tables.open_file(os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.hdf5'), mode='a')
        rad = h5file_phase.root.radius
        rad_vel = h5file_phase.root.radial_velocity
else:
    j = 0
    if args.slice2D:
        delta2D = np.zeros((args.grid_size,)*2, dtype=np.float64)
    if args.phase_space_1D:
        h5file_phase = tables.open_file(os.path.join(phasedir, f'phase-space_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{args.max_halos:d}.hdf5'), mode='w')
        atom = tables.Float64Atom()
        rad = h5file_phase.create_earray(h5file_phase.root, 'radius', atom, shape=(0,))
        rad_vel = h5file_phase.create_earray(h5file_phase.root, 'radial_velocity', atom, shape=(0,))

print('\n Starting with first halo')

for h in halos_this_step.index:
    t_now1 = time()

    halo_cen = halos_this_step[['x(17)','y(18)','z(19)']].loc[h].to_numpy()
    region = Region('cube', cen=halo_cen,side=L_cube_focus,box_size=snap.box_size)
    # region = Region('sphere', cen=halo_cen,rad=R_sphere_focus,box_size=snap.box_size)
    select_index = region.selectPrtcl(posd)

    print('\n particles selected in a covering region')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)

    posd_shifted = region.shift_origin(posd[select_index])
    # posd_focus =

    print('\n particle positions shifted to origin at halo centre')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)


    if args.align:
        rot_vec = halos_this_step[['A[x](45)','A[y](46)','A[z](47)']].loc[h].to_numpy()
        if not np.isclose(rot_vec,0).all():
            posd_focus = Transform.rotate(posd_shifted, rot_vec)
        else:
            print("\n Can\'t rotate for this halo, probably spherical")
            posd_focus = posd_shifted.copy()
    else:
        posd_focus = posd_shifted.copy()
            

    print('\n particle positions rotated')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)

    posd_focus += L_cube/2
    posd_cube = posd_focus[np.all((posd_focus>0) & (posd_focus<L_cube), axis=1)]
     
    # posd_cube = posd_focus[np.all((posd_focus>-L_cube/2) & (posd_focus<L_cube/2), axis=1)]
    # posd_cube += L_cube/2

    print('\n particles selected for the grid')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)
     

    particle_grid = assign_density(posd_cube, L_cube, args.grid_size, scheme=args.scheme, overdensity=False)
    
    print('\n density assignment is done')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)

    # pdb.set_trace()
    # print('start debug') 

    grid2D = project_to_slice(particle_grid, L_cube, axis=2, around_position=(L_cube/2,)*3, thick=slice_thickness)

    delta2D *= j
    delta2D += (grid2D / mean_dens) - 1
    j+=1
    delta2D /= j
    
    print('\n 2d projected density is obtained')
    t_bef1, t_now1 = t_now1, time()
    print(t_now1-t_bef1)
     
    if args.slice2D:
        slicedir = os.path.join(outdir,'slice2D')
        if not args.align:
            slicedir = os.path.join(slicedir, 'unaligned')
        os.makedirs(slicedir, exist_ok=True)
        np.save(os.path.join(slicedir, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.npy'), delta2D)
        with open(os.path.join(slicedir, f'slice_{args.snap_i:03d}_1by{args.downsample:d}_{args.M_around:.1e}_{max_halos_total:d}.meta'), 'w') as metafile:
            dict = {'N_stack':j, 'L_cube':L_cube, 'R_vir':R_vir, 'R_vir_root':R_vir_root, 'slice_thickness':slice_thickness}
            json.dump(dict,metafile, indent=True)
            # file.write(i)

    if args.phase_space_1D:
        rad_j = np.linalg.norm(posd_shifted - L_cube/2, axis=1)
        rad_vel_j = (veld[select_index] * posd_shifted).sum(axis=1) / rad_j
        rad.append(rad_j)
        rad_vel.append(rad_vel_j)

    # print('time at each step', t1, t2, t3, t4, t5, t6, t7)
    print('\n {0} number of halo-centric images stacked at snapshot {1}'.format(j, args.snap_i))
    t_bef, t_now = t_now, time()
    print('total time per halo', t_now-t_bef)
    
    sys.stdout.flush()
    # pdb.set_trace()



del posd
gc.collect()

if args.phase_space_1D:
    h5file_phase.close()



print('\n density assigned to grid around halos for snapshot {0:03d}'.format(args.snap_i))



# delta = particle_grid 

# if args.slice2D:
#     slicedir = os.path.join(outdir,'slice2D')
#     os.makedirs(slicedir, exist_ok=True)
#     np.save(os.path.join(slicedir, 'slice_{0:03d}_1by{1:d}.npy'.format(args.snap_i, args.downsample) ), delta2D)



# pdb.set_trace()











