import numpy as np
from numpy.linalg import matrix_rank

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import hankel

from functools import partial, lru_cache, reduce
from tqdm.autonotebook import tqdm

from .optimized import *

from sklearn.utils.extmath import randomized_svd


def ts_vector_to_trajectory_matrix(timeseries, L, K):
    hankelized = hankel(timeseries, np.zeros(L)).T
    hankelized = hankelized[:, :K]
    return hankelized


def ts_matrix_to_trajectory_matrix(timeseries, L, K):
    '''Forulation for V-MSSA (vertical stack)
    https://www.researchgate.net/publication/263870252_Multivariate_singular_spectrum_analysis_A_general_view_and_new_vector_forecasting_approach
    '''
    N, P = timeseries.shape

    trajectory_matrix = [
        ts_vector_to_trajectory_matrix(timeseries[:, p], L, K)
        for p in range(P)
    ]

    trajectory_matrix = np.concatenate(trajectory_matrix, axis=0)
    return trajectory_matrix


def decompose_trajectory_matrix(trajectory_matrix, K, svd_method='randomized'):
    # calculate S matrix
    # https://arxiv.org/pdf/1309.5050.pdf
    S = np.dot(trajectory_matrix, trajectory_matrix.T)

    # Perform SVD on S
    if svd_method == 'randomized':
        U, s, V = randomized_svd(S, K)
    elif svd_method == 'exact':
        U, s, V = np.linalg.svd(S)

    # Valid rank is only where eigenvalues > 0
    rank = np.sum(s > 0)

    # singular values are the square root of the eigenvalues
    s = np.sqrt(s)

    return U, s, V, rank


def sv_to_explained_variance_ratio(singular_values, N):
    # Calculation taken from sklearn. See:
    # https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/decomposition/pca.py
    eigenvalues = singular_values ** 2
    explained_variance = eigenvalues / (N - 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    return explained_variance, explained_variance_ratio


def singular_value_hard_threshold(singular_values,
                                  rank=None,
                                  threshold=2.858):
    rank = len(singular_values) if rank is None else rank

    # Singular Value Hard Thresholding
    # This is a threshold on the rank/singular values based on the findings
    # in this paper:
    # https://arxiv.org/pdf/1305.5870.pdf
    # We assume the noise is not known, and so the thresholding value is
    # determined by the data (See section D)
    median_sv = np.median(singular_values[:rank])
    sv_threshold = threshold * median_sv
    adjusted_rank = np.sum(singular_values >= sv_threshold)
    return adjusted_rank
