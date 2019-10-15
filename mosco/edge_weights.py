'''
Functions for computing nearest neighbor weights as described in:
Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data representation.
Neural Computation, 15:6, 1373-1396.


Two methods for determining connectivity are given:
    1. Radius: two points u,v are connected if and only if ||u - v|| < radius
    2. Nearest neighbors: two points u,v are connected if and only if u is one of v's k nearest neighbors or vice versa.

Two methods for determining weights:
    1. Binary weights (1 if connected, 0 otherwise).
    2. Heat kernel: if u and v are connected, weight = e^{-||u - v||^2 / t}, otherwise the weight is zero.

We implement the 2*2=4 variants.
'''
import numpy as np
import sklearn.neighbors
import scipy

def radius_binary(X, radius):
    return sklearn.neighbors.radius_neighbors_graph(X, radius, mode='connectivity', include_self=False)

def radius_distance(X, radius):
    return sklearn.neighbors.radius_neighbors_graph(X, radius, mode='distance', include_self=False)

def radius_heat_kernel(X, radius, t):
    neighbor_distances = radius_distance(X, radius)
    neighbor_distances.data = np.e**(-neighbor_distances.data**2/t) # Work only on the nonzero elements of the sparse matrix
    return neighbor_distances

def csr_iterate_keys(m):
    (height, width) = m.shape
    for i in xrange(height):
        for j in m[i].indices:
            yield (i,j)

def csr_has_key(m, (i,j)):
    return j in m[i].indices

def dok_update_min(dok, location, value):
    if dok.has_key(location):
        dok[location] = min(dok[location], value)
    else:
        dok[location] = value

def symmetrize_csr_matrix(m):
    (n,n1) = m.shape
    assert n == n1
    dok = scipy.sparse.dok_matrix(m.copy())
    for (i,j) in csr_iterate_keys(m):
        dok_update_min(dok, (i,j), m[i,j])
        dok_update_min(dok, (j,i), m[i,j])
    return dok.tocsr()

def knn_binary(X, k):
    (n,p) = X.shape
    assert k <= n-1
    knn_distances = sklearn.neighbors.kneighbors_graph(X, k, mode='connectivity', include_self=False)
    return symmetrize_csr_matrix(knn_distances)

def knn_distance(X, k):
    (n,p) = X.shape
    assert k <= n-1
    knn_distances = sklearn.neighbors.kneighbors_graph(X, k, mode='distance', include_self=False)
    result = symmetrize_csr_matrix(knn_distances)
    return result

def knn_heat_kernel(X, k, t):
    (n,p) = X.shape
    assert k <= n-1
    knn_distances = sklearn.neighbors.kneighbors_graph(X, k, mode='distance', include_self=False)
    symmetrized_knn_distances = symmetrize_csr_matrix(knn_distances)
    symmetrized_knn_distances.data = np.e**(-symmetrized_knn_distances.data**2/t)
    return symmetrized_knn_distances


