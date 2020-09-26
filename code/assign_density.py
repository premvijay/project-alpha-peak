import numpy as np
import pandas as pd

import os
import sys
from time import time

from gadget_tools import Snapshot, read_positions_all_files
from pm_tools import assign_density, project_to_slice
from field_tools import compute_power_spec

import socket
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(
    description='Assign density and compute power spectrum.',
    usage= 'python assign_density.py')

parser.add_argument('--snapdir', type=str, help='Directory containing the snapshot binaries')
parser.add_argument('--range', type=int, nargs=3, help='Range of snapshots to choose')

parser.add_argument('--scheme', type=str, help='Scheme for assigning particles to grid')
parser.add_argument('--grid-size', type=int, help='Grid size : number of cells along each direction')

parser.add_argument('--Pk', action='store_true', help='Compute and save power spectrum')
parser.add_argument('--slice2D', action='store_true', help='Compute and save 2D projected slices')

parser.add_argument('--outdir', type=str, help='Directory to save the requested output')

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ', rank, 'Hostname is', socket.gethostname() )

snapshots_to_run = np.arange(args.range[0],args.range[1],args.range[2], dtype=int)

snapshot_number_this_process = snapshots_to_run[rank]
# snapshot_number = sys.argv[2]

t_now = time()
print('\n Starting to read snapshots binaries')

filename_prefix = '/snapshot_{0:03d}'.format(snapshot_number_this_process)
filepath_prefix = args.snapdir + filename_prefix

filepath = filepath_prefix + '.0'
print(filepath)

snap = Snapshot()
snap.from_binary(filepath)

posd = read_positions_all_files(filepath_prefix)


delta = assign_density(posd, snap.box_size, args.grid_size, scheme=args.scheme)
del posd

if args.slice2D:
    mmhpos = (48.25266, 166.29897, 98.36325)
    delta_slice = project_to_slice(delta, snap.box_size, axis=2, around_position=mmhpos, thick=10)
    delta_slice.save(args.outdir+'/slice2D/'+'/slice_{0:03d}'.format(snapshot_number_this_process))

if args.Pk:
    power_spec = compute_power_spec(delta,snap.box_size)
    filepath = args.outdir+'/power_spectrum/'+'Pk_{0:03d}'.format(snapshot_number_this_process)
    power_spec.to_csv(filepath, sep='\t', index=False, 
                            float_format='%.8e', header=['k (h/Mpc)', 'P(k) (Mpc/h)^3'])




