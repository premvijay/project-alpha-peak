from . import particles_to_grid
import numpy as np

def assign_density(posd, box_size, grid_size = 512, scheme='CIC'):
    posd *= grid_size / box_size     # This happens without using extra memory
    posd_scaled_to_grid_units = posd   

    if scheme =='NGP':
        particle_grid = particles_to_grid.assign_ngp(posd_scaled_to_grid_units, grid_size)
    elif scheme=='CIC':
        particle_grid = particles_to_grid.assign_cic(posd_scaled_to_grid_units, grid_size)
    elif scheme=='TSC':
        particle_grid = particles_to_grid.assign_tsc(posd_scaled_to_grid_units, grid_size)
    else:
        raise NotImplementedError('Requested scheme is unknown or not yet implemented')
    density_grid = particle_grid
    overdensity_grid = density_grid / (np.mean(density_grid)+1e-5) - 1
    # print(density_grid)
    return overdensity_grid#, particle_grid
    