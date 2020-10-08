import numpy as np
import pandas as pd

import os
import sys
import pickle
from time import time, sleep

from gadget_tools import Snapshot, read_positions_all_files
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

parser.add_argument('--scheme', type=str, default='CIC',
                help='Scheme for assigning particles to grid')

parser.add_argument('--grid_size', type=int, default=512,
                help='Grid size : number of cells along each direction')

parser.add_argument('--align', action='store_true', help='Align and then stack')

parser.add_argument('--slice2D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--outdir', type=str, default='/scratch/cprem/sims',
                help='Directory to save the requested output')

args = parser.parse_args()

# simdir = '/scratch/aseem/sims' if args.simdir is None else args.simdir
# simname = 'bdm_cdm1024' if args.simname is None else args.simname
# outdir = '/scratch/cprem/sims' if args.outdir is None else args.outdir

# scheme = 'CIC' if args.scheme is None else args.scheme
# grid_size = 512 if args.grid_size is None else args.grid_size

snapdir = os.path.join(args.simdir, args.simname, args.rundir)
halosfile = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', 'halos_list/halos_select')

outdir = os.path.join(args.outdir, args.simname, args.rundir, 'halo_centric', args.scheme, '{0:d}'.format(args.grid_size) )
os.makedirs(outdir, exist_ok=True)

print('Hostname is', socket.gethostname(), flush=True )

t_now = time()
print('\n Starting to read snapshots binaries', flush=True)

filename_prefix = 'snapshot_{0:03d}'.format(args.snap_i)
filepath_prefix = os.path.join(snapdir, filename_prefix)

posd = read_positions_all_files(filepath_prefix)

# posd = posd[:10000]

print('\n Particle positions read from all binaries in the snapshot', flush=True)
t_bef, t_now = t_now, time()
print(t_now-t_bef, flush=True)

filepath = filepath_prefix + '.0'
print(filepath, flush=True)
snap = Snapshot()
snap.from_binary(filepath)






halos = pd.read_csv(halosfile, sep='\t', engine='c', index_col='id(1)')
halos_this_step = halos[halos['Snap_num(31)']==args.snap_i]
halos_root = halos[halos['Snap_num(31)']==200]

# choose cube size


L_cube = 5 * halos_root['rvir(11)'].mean() / 1e3
R_sphere_focus = np.sqrt(3)/2 * L_cube * (1+1e-2)
L_cube_focus = np.sqrt(3) * L_cube * (1+1e-2)

grid2D = np.zeros((args.grid_size,)*2, dtype=np.float64)

for h in halos_this_step.index:
    halo_cen = halos_this_step[['x(17)','y(18)','z(19)']].loc[h].to_numpy()
    posd_focus = Region('cube', cen=halo_cen,side=L_cube_focus,box_size=snap.box_size).selectPrtcl(posd, shift_origin=True)
    # posd_sphere = Region('sphere', cen=halo_cen,rad=R_sphere_focus,box_size=snap.box_size).selectPrtcl(posd)

    if args.align:
        posd_focus = Transform.rotate(posd_focus, rot_vec = halos_this_step[['A[x](45)','A[y](46)','A[z](47)']].loc[h].to_numpy())

    posd_focus += L_cube/2
    posd_cube = posd_focus[np.all((posd_focus>0) & (posd_focus<L_cube), axis=1)]

    particle_grid = assign_density(posd_cube, L_cube, args.grid_size, scheme=args.scheme, overdensity=False)
    
    print(particle_grid.shape, flush=True)

    # delta2D *= h
    grid2D += project_to_slice(particle_grid, L_cube, axis=2, around_position=(L_cube/2,)*3, thick=0.5)
    # delta2D /= (h+1)

    # pdb.set_trace()



mean_dens = posd.shape[0]/ (args.grid_size * snap.box_size / L_cube)**3

del posd
gc.collect()

delta2D = grid2D / len(halos_this_step.index) / mean_dens

print('\n density assigned to grid around halos for snapshot {0:03d}'.format(args.snap_i), flush=True)



# delta = particle_grid 

if slice2D:
    slicedir = os.path.join(outdir,'slice2D')
    os.makedirs(slicedir, exist_ok=True)
    np.save(os.path.join(slicedir, 'slice_{0:03d}.npy'.format(args.snap_i) ), delta2D)



# pdb.set_trace()











