import numpy as np
import matplotlib.pyplot as plt
import h5py

def propagate(wavefront,sample_to_object_distance,source_to_sample_distance=0,wavelength = 0,type='fourier'):
    
    if type == 'fourier':
        if z > 0:
            return np.fft.fftshift(np.fft.fft2(wavefront))
        elif z==0:
            return wavefront
        else:
            return np.fft.fftshift(np.fft.ifft2(wavefront))
            
    elif type == 'fresnel':

        if wavelength == 0:
            sys.exit('Please select a proper value for wavelength')
    
        z0 = sample_to_object_distance
        z = source_to_sample_distance
    
        if z0 != 0:
            M = (z+z_0)/z_0
        else:
            M=1
    
        wavevector = 2*np.pi/wavelength
        A = np.exp(1j*wavevector*distance)
        kernel = 1
        return A*np.fft.ifft2(kernel*np.fft.fftshift(np.fft.fft2(wavefront)))
    else:
        sys.exit('Please, select a valid propagator: fourier or fresnel') 


def read_cxi(path):
    data = h5py.File(path,'r')
    diffraction_patterns = data['entry_1/data_1/data'][:]
    positions = data['entry_1/data_1/translation'][:][:,0:2]
    pixel_size = data['entry_1/instrument_1/detector_1/x_pixel_size'][()]
    distance = data['entry_1/instrument_1/detector_1/distance'][()]
    energy = data['entry_1/instrument_1/source_1/energy'][()]

    one_kev_in_joules = 1.602e-16
    energy = energy / one_kev_in_joules
    
    return diffraction_patterns, positions, energy, distance, pixel_size
