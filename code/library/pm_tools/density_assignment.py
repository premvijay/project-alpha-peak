# To use this run compile script

from . import particles_to_grid
import numpy as np

def assign_density(posd, box_size, grid_size = 512, scheme='CIC', shift=0, overdensity=True):
    # posd *= grid_size / box_size    # This happens without using extra memory
    # posd += shift
    posd_scaled_to_grid_units = posd * grid_size / box_size + shift
    # posd_scaled_to_grid_units += shift

    if scheme =='NGP':
        particle_grid = particles_to_grid.assign_ngp(posd_scaled_to_grid_units, grid_size)
    elif scheme=='CIC':
        particle_grid = particles_to_grid.assign_cic(posd_scaled_to_grid_units, grid_size)
    elif scheme=='TSC':
        particle_grid = particles_to_grid.assign_tsc(posd_scaled_to_grid_units, grid_size)
    else:
        raise NotImplementedError('Requested scheme is unknown or not yet implemented')
    
    # print(density_grid)
    if overdensity:
        density_grid = particle_grid
        overdensity_grid = density_grid / (np.mean(density_grid)+1e-5) - 1
        return overdensity_grid
    else:
        return particle_grid
    