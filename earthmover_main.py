import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import wavetransform

import particle
import laplacian
import utils
import l2vsemd
from sklearn.metrics import pairwise_distances

BASEDIR = "./"

PARTICLE_DIR = BASEDIR

COMPUTATIONS_DIR = os.path.join(BASEDIR, 'computations')
COMPUTATION_DIR_NOISELESS = os.path.join(COMPUTATIONS_DIR, 'Noiseless Data/')
COMPUTATION_DIR_NOISY = os.path.join(COMPUTATIONS_DIR, 'STD = 0.01644027/')

#RAW_DATA_NOISELESS = os.path.join(COMPUTATION_DIR_NOISELESS, 'Raw Data/raw_data_800.npy')
#RAW_DATA_NOISY = os.path.join(COMPUTATION_DIR_NOISY, 'Raw Data/raw_data_800.npy')
RAW_DATA_NOISELESS = os.path.join(COMPUTATION_DIR_NOISELESS, 'Raw Data/raw_data_800.npy')
RAW_DATA_NOISY = os.path.join(COMPUTATION_DIR_NOISY, 'Raw Data/raw_data_800.npy')

FIGURES_DIR = os.path.join(BASEDIR, 'figures')

N_RANGE = [25,50,100,200,400,800]

SEED = 2019
MARKERSIZE = 75
MARKEREDGEWIDTH = 0.5 
DPI = 300
EXTENSION = 'eps'


def mkdir_computation_dirs():
    for subdir in ['Noiseless Data', 'STD = 0.01644027']:
        for subsubdir in ['Angles', 'Colors', 'Euclidean Distance Matrices', 'Raw Data', 'Times', 'WEMD Distance Matrices']:
            dirname = os.path.join(COMPUTATIONS_DIR, subdir, subsubdir)
            print('Making directory:', dirname)
            os.makedirs(dirname, exist_ok=True)



def generate_datasets(ns,std,particle_dir,computation_dir):
    print(f'Generating dataset: (std={std} computation_dir={computation_dir})')
    np.random.seed(SEED)
    for n in ns:
        print(f'    n={n}',)
        data,count,angles,colors = particle.generate_rotated_dataset(n, particle_dir, std)
        np.save(os.path.join(computation_dir, f"Raw Data/raw_data_{n}"), data)
        np.save(os.path.join(computation_dir, f"Angles/angles_{n}"), angles)
        np.save(os.path.join(computation_dir, f"Colors/colors_{n}"), colors)


def dist_matrix(data,metric,std,computation_dir):
    if metric == 'euclidean':
        new_data = []
        for item in data:
            new_data.append(item.flatten())
        start = time.time()
        dist = pairwise_distances(new_data,metric='euclidean')
        end = time.time()
        elapsed = end-start
        np.save(os.path.join(computation_dir, f'Euclidean Distance Matrices/dist_matrix_{metric}_{len(data)}'),dist)
        return(elapsed)
        
    elif metric == 'emd':
        start = time.time()
        waves = wavetransform.wave_transform_data(data)
        end = time.time()
        wave_elapsed = end-start
        start = time.time()
        dist = pairwise_distances(waves,metric='manhattan')
        end = time.time()
        elapsed = end-start
        np.save(os.path.join(computation_dir, f'WEMD Distance Matrices/dist_matrix_{metric}_{len(data)}'), dist)
        return(wave_elapsed,elapsed)
        
        
def calculate_distances(ns,std,computation_dir):
    print(f'Computing distances: (std={std} computation_dir={computation_dir})')
    euc_dist_times = []
    emd_dist_times = []
    wave_trans_times = []
    for n in ns:
        print(f'    n={n}',)
        data = np.load(computation_dir+f"Raw Data/raw_data_"+str(n)+".npy")
        euc_elapsed = dist_matrix(data,'euclidean',std,computation_dir)
        wave_elapsed,emd_elapsed = dist_matrix(data,'emd',std,computation_dir)
        euc_dist_times.append(euc_elapsed)
        emd_dist_times.append(emd_elapsed)
        wave_trans_times.append(wave_elapsed)

    np.save(os.path.join(computation_dir, f"Times/euc_dist_times_std={std}_{SEED}.npy"), euc_dist_times)
    np.save(os.path.join(computation_dir, f"Times/emd_dist_times_std={std}_{SEED}.npy"), emd_dist_times)
    np.save(os.path.join(computation_dir, f"Times/wave_trans_times_std={std}_{SEED}.npy"), wave_trans_times)


def initial_calculations(ns=N_RANGE):
    mkdir_computation_dirs()

    particle_dir = PARTICLE_DIR
    
    std = 0
    computation_dir = COMPUTATION_DIR_NOISELESS
    generate_datasets(ns, std, particle_dir, computation_dir)
    calculate_distances(ns, std, computation_dir)
    
    std = 0.01644027 
    computation_dir = COMPUTATION_DIR_NOISY
    generate_datasets(ns, std, particle_dir, computation_dir)
    calculate_distances(ns, std, computation_dir)
    
    
def mkdir_figures_dirs():
    for subdir in ['EMD-embeddings-CoifmanLafon', 'Euclidean-embeddings-CoifmanLafon']:
        dirname = os.path.join(FIGURES_DIR, subdir)
        print('Making directory:', dirname)
        os.makedirs(dirname, exist_ok=True)


def leading_eigenvectors(L_normalized, n_eigenvectors):
    assert utils.is_symmetric(L_normalized)
    assert L_normalized.dtype == float

    (eigvals, eigvecs) = scipy.sparse.linalg.eigs(L_normalized, n_eigenvectors, which='SM')
    return (eigvals.real, eigvecs.real.transpose())


def gaussian_kernel_distance_matrix(distance_matrix, sigma):
    assert distance_matrix.ndim == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]

    m = np.exp(-distance_matrix.astype(float)**2/(2*sigma*2))
    np.fill_diagonal(m, 0)

    return m


def plot_euclidean_embedding(computation_dir, n, sigma, markersize):
    WL2 = np.load(os.path.join(computation_dir, f'Euclidean Distance Matrices/dist_matrix_euclidean_{n}.npy'))
    colors = np.load(os.path.join(computation_dir, f'Colors/colors_{n}.npy'))
    W = gaussian_kernel_distance_matrix((WL2+WL2.T)/2.0, sigma)
    L = laplacian.geometric(W)
    (eigvals, eigvecs) = np.linalg.eig(L)
    eigvecs = np.array(eigvecs)
    plt.figure(figsize=(1.688, 1.688), dpi=DPI, frameon=False)
    plt.axis('off')
    plt.scatter(eigvecs[:,1], eigvecs[:,2], s=markersize, c=colors, edgecolors='black', linewidth=MARKEREDGEWIDTH)


def plot_wemd_embedding(computation_dir, n, sigma, markersize):
    Wemd = np.load(os.path.join(computation_dir, f'WEMD Distance Matrices/dist_matrix_emd_{n}.npy'))
    colors = np.load(os.path.join(computation_dir, f'Colors/colors_{n}.npy'))
    W = gaussian_kernel_distance_matrix((Wemd+Wemd.T)/2.0, sigma)
    L = laplacian.geometric(W)
    (eigvals, eigvecs) = np.linalg.eig(L)
    eigvecs = np.array(eigvecs)
    plt.figure(figsize=(1.688, 1.688), dpi=DPI, frameon=False)
    plt.axis('off')
    plt.scatter(eigvecs[:,1], eigvecs[:,2], s=markersize, c=colors, edgecolors='black', linewidth=MARKEREDGEWIDTH)


def figure_euclidean_embeddings_noiseless():
    for (n,sigma) in [(25, 5), (50, 3.4), (100, 3.3), (200, 3.2), (400, 2.3), (800, 1.8)]:
        if n not in N_RANGE:
            print(f'Skipping {n}')
            continue
        plot_euclidean_embedding(COMPUTATION_DIR_NOISELESS, n, sigma, MARKERSIZE)
        fn = os.path.join(FIGURES_DIR, f'Euclidean-embeddings-CoifmanLafon/euclidean_embedding_CL_{n}_noiseless_{SEED}.{EXTENSION}')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_wemd_embeddings_noiseless():
    SIGMA = 100000
    for n in N_RANGE:
        plot_wemd_embedding(COMPUTATION_DIR_NOISELESS, n, SIGMA, MARKERSIZE)
        fn = os.path.join(FIGURES_DIR, f'EMD-embeddings-CoifmanLafon/emd_embedding_CL_{n}_noiseless_{SEED}.{EXTENSION}')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_euclidean_embeddings_noisy():
    for (n,sigma) in [(25, 12), (50, 10), (100, 3.4), (200, 3.4), (400, 3.0), (800, 2.3)]:
        if n not in N_RANGE:
            print(f'Skipping {n}')
            continue
        plot_euclidean_embedding(COMPUTATION_DIR_NOISY, n, sigma, MARKERSIZE)
        fn = os.path.join(FIGURES_DIR, f'Euclidean-embeddings-CoifmanLafon/euclidean_embedding_CL_{n}_sigma0_01644_{SEED}.{EXTENSION}')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_wemd_embeddings_noisy():
    SIGMA = 100000
    for n in N_RANGE:
        plot_wemd_embedding(COMPUTATION_DIR_NOISY, n, SIGMA, MARKERSIZE)
        fn = os.path.join(FIGURES_DIR, f'EMD-embeddings-CoifmanLafon/emd_embedding_CL_{n}_sigma0_01644_{SEED}.{EXTENSION}')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_rotor_slice():
        VMIN =-0.06 
        VMAX = 0.150
        m = np.load(RAW_DATA_NOISELESS)
        plt.figure()
        plt.imshow(m[0,25,::-1,:], vmin=VMIN, vmax=VMAX)
        fn = os.path.join(FIGURES_DIR, 'slice_noiseless.png')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)

        m = np.load(RAW_DATA_NOISY)
        plt.figure()
        plt.imshow(m[0,25,::-1,:], vmin=VMIN, vmax=VMAX)
        fn = os.path.join(FIGURES_DIR, 'slice_noisy.png')
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)



def figure_all(): #Use to generate all data-based figures found in the paper
    mkdir_figures_dirs()
    print('    Generating euclidean embeddings (noiseless)')
    figure_euclidean_embeddings_noiseless()
    print('    Generating WEMD embeddings (noiseless)')
    figure_wemd_embeddings_noiseless()
    print('    Generating euclidean embeddings (noisy)')
    figure_euclidean_embeddings_noisy()
    print('    Generating WEMD embeddings (noisy)')
    figure_wemd_embeddings_noisy()
    print('    Saving rotor slices')
    figure_rotor_slice()
    print('    Computing and saving WEMD vs Euclidean figure')
    l2vsemd.emd_vs_euclid(COMPUTATIONS_DIR, PARTICLE_DIR, FIGURES_DIR)
    
