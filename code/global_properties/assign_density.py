import numpy as np
import pandas as pd

import os
import sys
import pickle
from time import time, sleep

import tables

from gadget_tools import Snapshot, read_positions_all_files
from pm_tools import assign_density, project_to_slice
from field_tools import compute_power_spec

import socket
# from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(
    description='Assign density and compute power spectrum.',
    usage= 'python assign_density.py')

parser.add_argument('--simdir', default='/scratch/cprem/sims/', type=str, help='Directory path for all simulations')
parser.add_argument('--simname', type=str, help='Simulation directory name')
parser.add_argument('--rundir', type=str, default='r1', help='Directory name containing the snapshot binaries')

parser.add_argument('--snap_i', type=int, help='Snapshot index number')

parser.add_argument('--scheme', type=str, help='Scheme for assigning particles to grid')
parser.add_argument('--grid_size', type=int, help='Grid size : number of cells along each direction')

parser.add_argument('--Pk', action='store_true', help='Compute and save power spectrum')
parser.add_argument('--interlace', action='store_true', help='Do interlacing for power spectrum')

parser.add_argument('--slice2D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--outdir', type=str, default='/scratch/cprem/sims/', help='Directory to save the requested output')

args = parser.parse_args()

snapdir = os.path.join(args.simdir, args.simname, args.rundir, 'snaps')
outdir = os.path.join(args.outdir, args.simname, args.rundir, 'global', args.scheme, '{0:d}'.format(args.grid_size) )
os.makedirs(outdir, exist_ok=True)

print('Hostname is', socket.gethostname() )

t_now = time()
print('\n Starting to read snapshots binaries')

def snapfilen_prefix(snapdirectory, snap_i):
    if os.path.exists(os.path.join(snapdir, f'snapdir_{snap_i:03d}')):
        return os.path.join(snapdir, 'snapdir_{0:03d}/snapshot_{0:03d}'.format(snap_i))
    else:
        return os.path.join(snapdir, 'snapshot_{0:03d}'.format(snap_i))

filepath_prefix = snapfilen_prefix(snapdir, args.snap_i)


# filename_prefix = 'snapdir_{0:03d}/snapshot_{0:03d}'.format(args.snap_i)
# filepath_prefix = os.path.join(snapdir, filename_prefix)

posd = read_positions_all_files(filepath_prefix)

# posd = posd[:10000]

print('\n Particle positions read from all binaries in the snapshot')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

filepath = filepath_prefix + '.0'
print(filepath)
snap = Snapshot()
snap.from_binary(filepath)

delta = assign_density(posd, snap.box_size, args.grid_size, scheme=args.scheme)
print('\n density assigned to main grid for snapshot {0:03d}'.format(args.snap_i))

if args.interlace:
    delta_shifted = assign_density(posd, snap.box_size, args.grid_size, scheme=args.scheme, shift=1/2)
    print('density assigned to shifted grid for snapshot {0:03d}'.format(args.snap_i))
else:
    delta_shifted = None


del posd


print('\n Density assigned for snapshot {0:03d} by scheme {1:s}'.format(args.snap_i, args.scheme))
t_bef, t_now = t_now, time()
print(t_now-t_bef)

# infodir = os.path.join(outdir,'info')
# os.makedirs(infodir, exist_ok=True)

# with open(os.path.join(infodir, 'header_{0:03d}.p'.format(args.snap_i) ),'wb') as headfile:
#     pickle.dump((snap),headfile)

dens_griddir = os.path.join(args.outdir, args.simname, args.rundir,'meshgrid')
os.makedirs(dens_griddir, exist_ok=True)
h5file = tables.open_file( os.path.join(dens_griddir, f'{args.scheme:s}_{args.grid_size:d}_{args.snap_i:03d}.hdf5'), 'w' )
grp_1 = h5file.create_group(h5file.root, "PartType1", "Dark matter")
h5file.create_array(grp_1, 'density', delta.astype(np.float32), 'Density assigned to grid dimensions')

if args.interlace:
    h5file.create_array(grp_1, 'density_shifted', delta_shifted.astype(np.float32), 'Density assigned to shifted grid dimensions')

h5file.close()

if args.slice2D:
    # mmhpos = (48.25266, 166.29897, 98.36325) #bdm_cdm
    # mmhpos = (48.31351, 166.24753, 98.45503000000001) #bdm_z
    delta_slice = project_to_slice(delta, snap.box_size, axis=2, around_position='centre', thick=10)
    slicedir = os.path.join(outdir,'slice2D')
    os.makedirs(slicedir, exist_ok=True)
    np.save(os.path.join(slicedir, 'slice_{0:03d}.npy'.format(args.snap_i) ), delta_slice)

if args.Pk:
    Pkdir = os.path.join(outdir,'power_spectrum')
    os.makedirs(Pkdir, exist_ok=True)
    power_spec = compute_power_spec(delta,snap.box_size, interlace_with_FX=None, Win_correct_scheme=args.scheme, grid_size=args.grid_size)
    filepath = os.path.join(Pkdir, 'Pk_{0:03d}.csv'.format(args.snap_i) )
    power_spec.to_csv(filepath, sep='\t', index=False, 
                            float_format='%.8e', header=['k (h/Mpc)', 'P(k) (Mpc/h)^3'])

    if args.interlace:
        power_spec_inlcd = compute_power_spec(delta,snap.box_size, interlace_with_FX=delta_shifted, Win_correct_scheme=args.scheme, grid_size=args.grid_size)
        filepath = os.path.join(Pkdir, 'Pk_interlaced_{0:03d}.csv'.format(args.snap_i) )
        power_spec_inlcd.to_csv(filepath, sep='\t', index=False, 
                            float_format='%.8e', header=['k (h/Mpc)', 'P(k) (Mpc/h)^3'])

        
    





