import scipy
import numpy as np
import matplotlib.pyplot as plt
import laplacian

COMPUTATION_DIR_CLEAN = '/Users/mosco/Dropbox/research/earthmover/computations/Max Level = 5/Noiseless Data/'
COMPUTATION_DIR_NOISY = '/Users/mosco/Dropbox/research/earthmover/computations/Max Level = 5/STD = 0.01644027/'
FIGURES_DIR = '/Users/mosco/Dropbox/research/earthmover/figures/'
SEED = 2019
MARKERSIZE = 75

#SIGMA_L2 = sorted(WL2[405])[5] #1.75567

def is_symmetric(A):
    return not boolmatrix_any(A != A.transpose())


def boolmatrix_any(A):
    assert A.dtype == np.bool

    if scipy.sparse.issparse(A):
        return A.nnz > 0
    else:
        return A.any()


def leading_eigenvectors(L_normalized, n_eigenvectors):
    assert is_symmetric(L_normalized)
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
    angles = np.load(computation_dir + f'Angles/angles_{n}.npy')
    WL2 = np.load(computation_dir + f'Euclidean Distance Matrices/dist_matrix_euclidean_{n}.npy')
    colors = np.load(computation_dir + f'Colors/colors_{n}.npy')
    W = gaussian_kernel_distance_matrix((WL2+WL2.T)/2.0, sigma)
    L = laplacian.geometric(W)
    (eigvals, eigvecs) = np.linalg.eig(L)
    eigvecs = np.array(eigvecs)
    plt.figure(figsize=(1.688, 1.688), dpi=600, frameon=False)
    plt.axis('off')
    plt.scatter(eigvecs[:,1], eigvecs[:,2], s=markersize, c=colors, edgecolors='black')


def plot_wemd_embedding(computation_dir, n, sigma, markersize):
    angles = np.load(computation_dir + f'Angles/angles_{n}.npy')
    Wemd = np.load(computation_dir + f'WEMD Distance Matrices/dist_matrix_emd_{n}.npy')
    colors = np.load(computation_dir + f'Colors/colors_{n}.npy')
    W = gaussian_kernel_distance_matrix((Wemd+Wemd.T)/2.0, sigma)
    L = laplacian.geometric(W)
    (eigvals, eigvecs) = np.linalg.eig(L)
    eigvecs = np.array(eigvecs)
    plt.figure(figsize=(1.688, 1.688), dpi=600, frameon=False)
    plt.axis('off')
    plt.scatter(eigvecs[:,1], eigvecs[:,2], s=markersize, c=colors, edgecolors='black')


def figure_euclidean_embeddings_noiseless():
    for (n,sigma) in [(25, 5), (50, 3.4), (100, 3.3), (200, 3.2), (400, 2.3), (800, 1.8)]:
        plot_euclidean_embedding(COMPUTATION_DIR_CLEAN, n, sigma, MARKERSIZE)
        fn = FIGURES_DIR + f'Euclidean-embeddings-CoifmanLafon/euclidean_embedding_CL_{n}_noiseless_{SEED}.pdf'
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_wemd_embeddings_noiseless():
    SIGMA = 100000
    for n in [25,50,100,200,400,800]:
        plot_wemd_embedding(COMPUTATION_DIR_CLEAN, n, SIGMA, MARKERSIZE)
        fn = FIGURES_DIR + f'EMD-embeddings-CoifmanLafon/emd_embedding_CL_{n}_noiseless_{SEED}.pdf'
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_euclidean_embeddings_noisy():
    for (n,sigma) in [(25, 12), (50, 10), (100, 3.4), (200, 3.4), (400, 3.0), (800, 2.3)]:
        plot_euclidean_embedding(COMPUTATION_DIR_NOISY, n, sigma, MARKERSIZE)
        fn = FIGURES_DIR + f'Euclidean-embeddings-CoifmanLafon/euclidean_embedding_CL_{n}_sigma0_01644_{SEED}.pdf'
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_wemd_embeddings_noisy():
    SIGMA = 100000
    #for n in [25,50,100,200,400,800]:
    for n in [25,50,100,200,400,800]:
        plot_wemd_embedding(COMPUTATION_DIR_NOISY, n, SIGMA, MARKERSIZE)
        fn = FIGURES_DIR + f'EMD-embeddings-CoifmanLafon/emd_embedding_CL_{n}_sigma0_01644_{SEED}.pdf'
        print('Saving', fn)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)


def figure_all():
    figure_euclidean_embeddings_noiseless()
    figure_wemd_embeddings_noiseless()
    figure_euclidean_embeddings_noisy()
    figure_wemd_embeddings_noisy()

