import numpy as np
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser(description='Find most massive in a given tree.',
usage= 'python find_most_massive_halo.py --treefile /scratch/aseem/halos/bdm_cdm1024/r1/out_200.trees')

parser.add_argument('--treefile', type=str)

args = parser.parse_args()


tree = pd.read_csv(args.treefile, sep=r'\s+', header =0, skiprows=list(range(1,58)))# comment='#')#, usecols = [0,1,2])

# tree = np.genfromtxt(filepath)

print(tree.head(), '\n\n', tree.columns, '\n\n', tree.info(),'\n\n', tree.memory_usage())


mmhi = tree['mvir(10)'].idxmax()

mmhpos = (tree['x(17)'].iloc[mmhi], tree['y(18)'].iloc[mmhi], tree['z(19)'].iloc[mmhi])

print('\n\n', mmhi, mmhpos)