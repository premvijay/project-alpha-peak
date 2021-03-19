import yt
import numpy as np
import yt.units as units
import pylab

L=150
N=256
i=2
rund='r2'

fname = f"/scratch/cprem/sims/L{L:d}_N{N:d}_Cp18_bar/{rund:s}/snaps/snapdir_{i:03d}/snapshot_{i:03d}.0.hdf5"

unit_base = {'UnitLength_in_cm'         : 3.08568e+24,
             'UnitMass_in_g'            :   1.989e+43,
             'UnitVelocity_in_cm_per_s' :      100000}

bbox_lim = L #Mpc

bbox = [[0,bbox_lim],
        [0,bbox_lim],
        [0,bbox_lim]]
 
ds = yt.load(fname, unit_base=unit_base)#, bounding_box=bbox)#, cosmology_parameters={})
ds.index

ad= ds.all_data()

yt.interactive_render(ds)