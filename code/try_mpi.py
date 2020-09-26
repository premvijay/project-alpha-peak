import subprocess
import socket
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(description='Assign density and compute power spectrum.')

parser.add_argument('--usedir', type=str)

args = parser.parse_args()

# command = 'hostname'
# process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ',rank, 'Hostname is', socket.gethostname() )
print(args.usedir)

