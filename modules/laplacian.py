import numpy as np
import scipy.sparse
import utils


def _is_legal_undirected_edge_weights_matrix(W):
    """edge_weights must be non-negative, symmetric and with a zero diagonal"""
    if utils.boolmatrix_any(W < 0):
        return False
    elif not utils.is_symmetric(W):
        return False
    elif utils.boolmatrix_any(W.diagonal() != 0):
        return False

    return True


def _get_degrees(W):
    return np.sum(W, axis=1, dtype=np.float)


def _compute_D(W):
    """Given a symmetric matrix of edge weights, compute the vertex degrees
    and place the result in a sparse diagonal matrix"""
    degrees = _get_degrees(W)
    return np.diag(degrees)
    #return scipy.sparse.spdiags(degrees, [0], n, n)


def combinatorial(W):
    assert _is_legal_undirected_edge_weights_matrix(W)

    D = _compute_D(W)
    return D-W


def normalized(W):
    assert _is_legal_undirected_edge_weights_matrix(W)

    degrees = _get_degrees(W)
    assert (degrees > 0).all()

    invsqrtdegrees = 1.0/np.sqrt(degrees)
    I = np.eye(W.shape[0])
    L_normalized = I - np.einsum('ij,i,j->ij', W, invsqrtdegrees, invsqrtdegrees)
    return (L_normalized + L_normalized.T)/2


def randomwalk(W):
    assert _is_legal_undirected_edge_weights_matrix(W)

    degrees = _get_degrees(W)
    assert (degrees > 0).all()
    invdegrees = 1.0/degrees

    I = scipy.sparse.identity(W.shape[0])
    return  I - W*invdegrees[:,np.newaxis]


def geometric(W):
    assert _is_legal_undirected_edge_weights_matrix(W)

    degrees = _get_degrees(W)
    assert (degrees > 0).all()
    invdegrees = 1.0/degrees

    W_new = np.einsum('ij,i,j->ij', W, invdegrees, invdegrees)

    # W_new should be almost symmetric, up to rounding errors. We resymmetrize it.
    W_new = (W_new + W_new.transpose())/2 
    return randomwalk(W_new)
