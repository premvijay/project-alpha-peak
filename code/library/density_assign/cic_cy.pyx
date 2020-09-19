import numpy as np
cimport numpy as np

cpdef np.ndarray[np.double_t, ndim=1] fn(int a):
    ar = np.array([a,a,a])
    return ar

