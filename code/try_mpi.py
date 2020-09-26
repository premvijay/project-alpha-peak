import subprocess
import socket
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(description='Assign density and compute power spectrum.')

parser.add_argument('--use_dir', type=str)
parser.add_argument('--range', type=int, nargs=3, help='start from snapshot number')

args = parser.parse_args()

# command = 'hostname'
# process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ',rank, 'Hostname is', socket.gethostname() )
print(args.use_dir)
print(np.arange(args.range[0],args.range[1],args.range[2], dtype=int))

