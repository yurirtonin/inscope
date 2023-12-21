import numpy as np
import h5py

path = "/Users/yuri/coding/data/siemens.h5"

data = h5py.File(path,'r')

DPs = data['exchange/data']
positions = data['metadata/probe_pos_px']
energy = data['metadata/energy_ev']
pixel_size = data['metadata/psize_cm']*1e-2

