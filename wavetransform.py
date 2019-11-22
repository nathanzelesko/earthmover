import numpy as np
import pywt
import utils

DIMENSION = 3


def wave_emd(p1,p2):
    p = np.asarray(p1)-np.asarray(p2)
    p = np.abs(p)
    emd = np.sum(p)
    return emd


def volume_to_wavelet_domain(volume, level, wavelet):
    """
    This function computes an embedding of non-negative 3D Numpy arrays such that the L_1 distance
    between the resulting embeddings is approximately equal to the Earthmover distance of the arrays.

    It implements the weighting scheme in Eq. (20) of the Technical report by Shirdhonkar, Sameer, and David W. Jacobs. "CAR-TR-1025 CS-TR-4908 UMIACS-TR-2008-06." (2008).
    """
    assert len(volume.shape) == DIMENSION

    volume_dwt = pywt.wavedecn(volume/volume.sum(), wavelet, mode='zero', level=level)

    detail_coefs = volume_dwt[1:]
    n_levels = len(detail_coefs)

    weighted_coefs = []
    for (j, details_level_j) in enumerate(volume_dwt[1:]):
        for coefs in details_level_j.values():
            multiplier = 2**((n_levels-1-j)*(1+(DIMENSION/2.0)))
            weighted_coefs.append(coefs.flatten()*multiplier)

    return np.concatenate(weighted_coefs)


def wave_transform_data(data):
    waves = []
    for image in data:
        waves.append(volume_to_wavelet_domain(image))
    return waves
