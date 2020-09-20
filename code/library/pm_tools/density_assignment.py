from . import particles_to_grid
import numpy as np

def assign_density(posd, box_size, grid_size = 512, scheme='CIC'):
    posd_scaled_to_grid_units = posd * grid_size / box_size
    if scheme =='NGP':
        grid = particles_to_grid.assign_ngp(posd_scaled_to_grid_units, grid_size)
    elif scheme=='CIC':
        grid = particles_to_grid.assign_cic(posd_scaled_to_grid_units, grid_size)
    elif scheme=='TSC':
        grid = particles_to_grid.assign_tsc(posd_scaled_to_grid_units, grid_size)
    else:
        raise NotImplementedError('Requested scheme is unknown or not yet implemented')
    density_grid = grid
    print(density_grid)
    overdensity_grid = density_grid / (np.mean(density_grid)+1e-5) - 1
    print(density_grid)
    return overdensity_grid, grid