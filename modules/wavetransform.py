import numpy as np
import pywt

def wave_emd(p1,p2):
    p = np.asarray(p1)-np.asarray(p2)
    p = np.abs(p)
    emd = np.sum(p)
    return emd

def volume_to_wavelet_domain(p0,l=5):
    p = p0.copy()
    m = np.sum(p)
    p = np.divide(p,m)
    
    wavelet = pywt.Wavelet('coif3')
    
    coeffs = pywt.wavedecn(p,wavelet,mode='zero',level=l)
    
    coeffs = coeffs[1:]
    vect = []
    for j in range(len(coeffs)):
        for entry in coeffs[-1-j]:
            flat = np.asarray(coeffs[-1-j][entry]).flatten(order='C')
            for item in flat:
                vect.append(item*(2**(j*(1+(3/2)))))
    return np.asarray(vect).flatten()


def wave_transform_data(data):
    waves = []
    for image in data:
        waves.append(volume_to_wavelet_domain(image))
    return waves
