# Python modules
import numpy as np
import h5py

# Internal modules
import misc

class POD:
    def __init__(self):
        self.DP = None # diffraction pattern
        self.obj = None # object
        self.probe = None
        self.positions = None
        self.energy = 0.0
        self.detector_pixel = 0.0
        self.detector_distance = 0.0
        self.source_distance = np.inf
        self.modes = 1
        self.object_pixel = 0.0
    
    def load_data(self,DP,obj,probe,positions,energy,detector_pixel,detector_distance,modes=1,source_distance=np.inf):
        self.DP = DP
        self.obj = obj
        self.probe = probe
        self.positions = positions
        self.energy = energy
        self.detector_distance = detector_distance
        self.detector_pixel = detector_pixel
        self.modes = modes
        self.source_distance = source_distance
        self.object_pixel = misc.get_object_pixel_size(self.energy,self.detector_distance,self.DP.shape,self.detector_pixel)

    def load_h5_data(self,path,conversion_func=None):
        h5_data = h5py.File(path,'r')
        self.DP = h5_data['/exchange/data'][()]
        self.energy = h5_data['metadata/energy_ev']
        self.detector_pixel = h5_data['metadata/psize_cm']
        self.positions = h5_data['metadata/probe_pos_px']
        self.wavelength = misc.energy_to_wavelength(self.energy)
        self.object_pixel = misc.get_object_pixel_size(self.energy,self.detector_distance,self.DP.shape,self.detector_pixel)

        if conversion_func is not None:
            self.positions = self.convert_positions_to_pixels(self,conversion_func,self.positions)

        self.obj = self.set_object_matrix(self.DP.shape,self.positions)

    def convert_positions_to_pixels(self,conversion_func,positions):
        return conversion_func(positions)
    
    def set_object_matrix(data_shape,positions):
        return 0