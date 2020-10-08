import numpy as np


def average_2D_slice(grid, box_size, axis, around_position, thick):
    start = (around_position[axis] - thick/2) * grid.shape[axis] // box_size
    stop = (around_position[axis] + thick/2) * grid.shape[axis] // box_size
    print(start,stop, flush=True)
    idx_range = np.arange(start,stop,dtype='int') % grid.shape[axis]
    idx_range_3D = tuple([slice(None)]* axis + [idx_range])
    print(idx_range_3D, flush=True)
    return grid[idx_range_3D].mean(axis=axis)