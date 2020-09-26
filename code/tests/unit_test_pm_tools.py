import numpy as np
import os
import sys
import pdb

# library_path = os.path.abspath(os.path.join('.'))
# print(library_path)
library_path = '/mnt/home/student/cprem/project-alpha-peak/code/library'
if library_path not in sys.path:
    raise Exception
    sys.path.append(library_path)



# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

from pm_tools import assign_density



posd = np.array([[2,2,2.5]])
delta = assign_density(posd, 4,4, scheme='TSC')
print(delta+1)



pdb.set_trace()










