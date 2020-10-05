import numpy as np
import pandas as pd

import os
import sys
import pickle
from time import time

from gadget_tools import Snapshot, read_positions_all_files
from pm_tools import assign_density, project_to_slice
from field_tools import compute_power_spec

import socket
# from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(
    description='Assign density and compute power spectrum.',
    usage= 'python assign_density.py')

parser.add_argument('--snapdir', type=str, help='Directory containing the snapshot binaries')
parser.add_argument('--snap_i', type=int, help='Snapshot index number')

parser.add_argument('--scheme', type=str, help='Scheme for assigning particles to grid')
parser.add_argument('--grid_size', type=int, help='Grid size : number of cells along each direction')

parser.add_argument('--Pk', action='store_true', help='Compute and save power spectrum')
parser.add_argument('--interlace', action='store_true', help='Do interlacing for power spectrum')

parser.add_argument('--slice2D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--outdir', type=str, help='Directory to save the requested output')

args = parser.parse_args()

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print('My rank is ', rank,)
print('Hostname is', socket.gethostname() )

# snapshots_to_run = np.arange(args.range[0],args.range[1],args.range[2], dtype=int)

# snapshot_number_this_process = snapshots_to_run[rank]
# snapshot_number = sys.argv[2]

t_now = time()
print('\n Starting to read snapshots binaries')

filename_prefix = '/snapshot_{0:03d}'.format(args.snap_i)
filepath_prefix = args.snapdir + filename_prefix

posd = read_positions_all_files(filepath_prefix)

posd = posd[:10000]

print('\n Particle positions read from all binaries in the snapshot')
t_bef, t_now = t_now, time()
print(t_now-t_bef)

filepath = filepath_prefix + '.0'
print(filepath)

snap = Snapshot()
snap.from_binary(filepath)

delta = assign_density(posd, snap.box_size, args.grid_size, scheme=args.scheme)
if args.interlace:
    delta_shifted = assign_density(posd, snap.box_size, args.grid_size, scheme=args.scheme, shift=1/2)
else:
    delta_shifted = None

del posd

print('\n Density assigned for snapshot {0:03d}'.format(args.snap_i))
t_bef, t_now = t_now, time()
print(t_now-t_bef)

with open(args.outdir+'/info/'+'/header_{0:03d}.p'.format(args.snap_i),'wb') as headfile:
    pickle.dump((snap),headfile)


if args.slice2D:
    mmhpos = (48.25266, 166.29897, 98.36325)
    delta_slice = project_to_slice(delta, snap.box_size, axis=2, around_position=mmhpos, thick=10)
    np.save(args.outdir+'/slice2D/'+'/slice_{0:03d}.npy'.format(args.snap_i), delta_slice)

if args.Pk:
    power_spec = compute_power_spec(delta,snap.box_size, interlace_with_FX=delta_shifted)
    filepath = args.outdir+'/power_spectrum/'+'Pk_{0:03d}.csv'.format(args.snap_i)
    power_spec.to_csv(filepath, sep='\t', index=False, 
                            float_format='%.8e', header=['k (h/Mpc)', 'P(k) (Mpc/h)^3'])





