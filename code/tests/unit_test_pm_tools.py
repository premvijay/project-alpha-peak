import numpy as np
import os
import sys
import pdb

library_path = os.path.abspath(os.path.join('.'))
print(library_path)
if library_path not in sys.path:
    sys.path.append(library_path)


# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

from library.pm_tools import assign_density



posd = np.array([[2,2,2.5]])
delta, particle_grid = assign_density(posd, 4,4, scheme='TSC')
print(particle_grid)



pdb.set_trace()










