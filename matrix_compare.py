import numpy as np
from scipy.linalg import norm
import itertools
import sas


# functions that actually compute matrix similarities
# def fidelity(Kx, Ky):
#     Kx = Kx + np.zeros(Kx.shape, dtype=complex)
#     Ky = Ky + np.zeros(Ky.shape, dtype=complex)
#     inner = ((Kx**0.5) @ Ky @ (Kx**0.5)) ** 0.5
#     return np.trace(inner)

def procrustes_size_shape_distance(X, Y):
    """compute the Procrustes size and shape distance between two matrices. NOT the Angular Procrustes distance"""
    assert X.shape[0] == Y.shape[0]
    centering = np.identity(X.shape[0]) - ((1/X.shape[0])*np.ones((X.shape[0], X.shape[0])))

    Σx = X.T @ centering @ X # results in shape columns x columns
    Σy = Y.T @ centering @ Y

    Σxy = X.T @ centering @ Y

    return np.sqrt(np.trace(Σx) + np.trace(Σy) - 2*norm(Σxy, ord='nuc'))


def riemannian_shape_distance(X, Y):
    """compute the Riemann shape distance between two matrices"""
    assert X.shape[0] == Y.shape[0]
    centering = np.identity(X.shape[0]) - ((1/X.shape[0])*np.ones((X.shape[0], X.shape[0])))

    Σx = X.T @ centering @ X # results in shape columns x columns
    Σy = Y.T @ centering @ Y

    Σxy = X.T @ centering @ Y

    return np.arccos(norm(Σxy, ord='nuc') / np.sqrt(np.trace(Σx)*np.trace(Σy)))


# def normalized_bures_similarity(X, Y):
#     assert X.shape[0] == Y.shape[0]
#     centering = np.identity(X.shape[0]) - ((1/X.shape[0])*np.ones((X.shape[0], X.shape[0])))
#     Kx = centering @ X @ X.T @ centering
#     Ky = centering @ Y @ Y.T @ centering
#     f = fidelity(X, Y)
#     return f / np.sqrt(np.trace(Kx)*np.trace(Ky))


# def bures_distance(Kx, Ky):
#     inner = np.trace(Kx) + np.trace(Ky) - 2*fidelity(Kx, Ky)
#     return np.sqrt(inner)


def linear_cka(X, Y):
    """compute the linear centralized kernel alignment between two matrices. Should be given as (sample,feature)"""
    assert X.shape[0] == Y.shape[0]
    centering = np.identity(X.shape[0]) - ((1/X.shape[0])*np.ones((X.shape[0], X.shape[0])))

    Kx = centering @ X @ X.T @ centering
    Ky = centering @ Y @ Y.T @ centering
    
    numer = np.trace(Kx @ Ky)
    denom = np.sqrt(np.trace(Kx @ Kx) * np.trace(Ky @ Ky))
    return numer / denom





#
# calling functions that handle cases of more than 2 input matrices
#
def similarity_compute(*args, f=linear_cka):
    """default function is linear CKA"""
    combos = itertools.combinations(args, 2)
    s = []
    for i, combo in enumerate(combos):
        s.append(f(*combo))
    return np.array(s)


def nuc_similarity(*args):
    combos = itertools.combinations(args, 2)
    s = []
    for i, combo in enumerate(combos):
        diff = combo[1] - combo[0]
        s.append(norm(diff, ord='nuc'))
    return np.array(s)



def frob_similarity(*args):
    combos = itertools.combinations(args, 2)
    s = []
    for i, combo in enumerate(combos):
        diff = combo[1] - combo[0]
        s.append(norm(diff, ord='fro'))
    return np.array(s)
    
