def phase_derivative_hilbert_transform(sinogram,pixel_size=1):
    """
    Calculate phase derivative to use it with backprojection (no filter!) 
    Pixel size in meters
    """
    g = np.exp(1j*sinogram) 
    grad_sinogram = np.angle(g * np.conj(np.roll(g,shift=1,axis=-1)))/pixel_size
    hilbert = np.imag(scipy.signal.hilbert(grad_sinogram,axis=-1)/(2*np.pi))
    return hilbert