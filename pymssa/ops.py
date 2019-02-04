import numpy as np
from numpy.linalg import matrix_rank

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import hankel
import scipy

from functools import partial, lru_cache, reduce
from tqdm.autonotebook import tqdm

from .optimized import *

from sklearn.utils.extmath import randomized_svd

import time

from sklearn.covariance import MinCovDet, LedoitWolf

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess


def ts_vector_to_trajectory_matrix(timeseries, L, K):
    hankelized = hankel(timeseries, np.zeros(L)).T
    hankelized = hankelized[:, :K]
    return hankelized


def ts_matrix_to_trajectory_matrices(timeseries, L, K):
    N, P = timeseries.shape

    trajectory_matrices = [
        ts_vector_to_trajectory_matrix(timeseries[:, p], L, K)
        for p in range(P)
    ]
    return trajectory_matrices


def batch_construct_elementary_matrix(left_singular_vectors,
                                      factor_vectors,
                                      singular_values,
                                      rank):
    print('construct U,s,V and reshape')
    lsv_r, lsv_c = left_singular_vectors[:, :rank].shape
    fv_r, fv_c = factor_vectors[:rank, :].shape

    U = left_singular_vectors[:, :rank].T.reshape((lsv_c, lsv_r, 1))
    V = factor_vectors[:rank, :].reshape((fv_r, 1, fv_c))
    s = singular_values[:rank].reshape((lsv_c, 1, 1))

    print('recon matrix')
    X = (U @ V) * s
    X = np.moveaxis(X, 0, -1)
    return X


def trajectory_matrix_dehankelization_indices(N, L, K):
    ts_indices = np.arange(N)
    hankelized_indices = hankel(ts_indices, np.zeros(L)).T
    hankelized_indices = hankelized_indices[:, :K].astype(int)
    return hankelized_indices


def dehankelization_indexers(indexer_matrix):
    row_indexer = np.repeat(np.arange(indexer_matrix.shape[0]), indexer_matrix.shape[1])
    col_indexer = indexer_matrix.ravel()
    return row_indexer, col_indexer


def indexer_dehankelization(elementary_matrix,
                            row_indexer,
                            col_indexer,
                            P,
                            N,
                            L):

    print('construct DH mat')
    R = elementary_matrix.shape[-1]
    dehankelization_matrix = np.zeros((P, L, N, R), dtype=float)
    dehankelization_matrix[:] = np.nan

    #for r in range(R):
    #    dehankelization_matrix[:, row_indexer, col_indexer, r] = elementary_matrix[:,:,:,r].reshape((P, -1))

    print('dehankelization insertion')
    dehankelization_matrix = batch_dehankelization_inserter(
        dehankelization_matrix,
        elementary_matrix,
        row_indexer,
        col_indexer,
        P,
        R
    )

    print('nanmean')
    dehankelized = np.nanmean(dehankelization_matrix, axis=1)
    return dehankelized


def elementary_matrix_by_timeseries(elementary_matrix,
                                    P,
                                    L,
                                    K):
    _N, _M, _r = elementary_matrix.shape

    if _N == (P * L):
        elementary_matrix = elementary_matrix.reshape((P, L, K, _r))

    elif _N == L:
        elementary_matrix = elementary_matrix.reshape((L, P, K, _r))
        elementary_matrix = np.moveaxis(elementary_matrix, 1, 0)

    return elementary_matrix


def indexer_component_reconstruction(trajectory_matrix,
                                     left_singular_vectors,
                                     factor_vectors,
                                     singular_values,
                                     rank,
                                     P,
                                     N,
                                     L,
                                     K):

    indexer_matrix = trajectory_matrix_dehankelization_indices(N, L, K)
    row_indexer, col_indexer = dehankelization_indexers(indexer_matrix)

    print('1. construct EM')
    elementary_matrix = batch_construct_elementary_matrix(
        left_singular_vectors,
        factor_vectors,
        singular_values,
        rank
    )
    print(elementary_matrix.shape)

    print('2. reshape EM')
    elementary_matrix = elementary_matrix_by_timeseries(
        elementary_matrix,
        P,
        L,
        K
    )
    print(elementary_matrix.shape)

    print('3. construct components')
    components = indexer_dehankelization(
        elementary_matrix,
        row_indexer,
        col_indexer,
        P,
        N,
        L
    )
    print(components.shape)

    return components


def incremental_component_reconstruction(left_singular_vectors,
                                         factor_vectors,
                                         singular_values,
                                         rank,
                                         P,
                                         N,
                                         L,
                                         K):

    components = np.zeros((P, N, rank))

    # indexer_matrix = trajectory_matrix_dehankelization_indices(N, L, K)
    # row_indexer, col_indexer = dehankelization_indexers(indexer_matrix)
    #
    # components = np.zeros((P, L, N, rank), dtype=float)
    # components[:] = np.nan

    start_t = time.time()
    components = incremental_component_reconstruction_inner(
        components,
        left_singular_vectors,
        factor_vectors,
        singular_values,
        P,
        L,
        K
    )
    end_t = time.time()
    print('Reconstruction elapsed time: {:.2f}'.format(end_t - start_t))

    return components


def vertically_stacked_trajectory_matrices(timeseries, L, K):
    '''Forulation for V-MSSA (vertical stack)
    https://www.researchgate.net/publication/263870252_Multivariate_singular_spectrum_analysis_A_general_view_and_new_vector_forecasting_approach
    '''
    trajectory_matrices = ts_matrix_to_trajectory_matrices(timeseries, L, K)
    trajectory_matrix = np.concatenate(trajectory_matrices, axis=0)
    return trajectory_matrix


def horizontally_stacked_trajectory_matrix(timeseries, L, K):
    '''Formulation for H-MSSA (horizontal stack)'''
    trajectory_matrices = ts_matrix_to_trajectory_matrices(timeseries, L, K)
    trajectory_matrix = np.concatenate(trajectory_matrices, axis=1)
    return trajectory_matrix


def decompose_trajectory_matrix(trajectory_matrix, K, svd_method='randomized'):
    # calculate S matrix
    # https://arxiv.org/pdf/1309.5050.pdf
    #S = np.dot(trajectory_matrix, trajectory_matrix.T)
    S = np.cov(trajectory_matrix)

    #print('Covariance estimation')
    #cov_est = MinCovDet(store_precision=False, support_fraction=0.85)
    #cov_est.fit(trajectory_matrix.T)

    #cov_est = LedoitWolf()
    #cov_est.fit(trajectory_matrix.T)

    #S = cov_est.covariance_

    # Perform SVD on S
    if svd_method == 'randomized':
        U, s, V = randomized_svd(S, K)
    elif svd_method == 'exact':
        U, s, V = np.linalg.svd(S)
    elif svd_method == 'robust':
        U, s, V = robust_svd(S, K)

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



def optimal_component_ordering(timeseries,
                               components):
    # Calculate residuals in terms as what is lost removing a component
    # from the full reconstruction.
    # Uses sum of squared error. Could be changed to another loss easily if
    # preferred.

    n_ts = timeseries.shape[1]
    n_cp = components.shape[2]

    optimal_orders = np.zeros((n_cp, n_ts))
    component_sses = np.zeros((n_ts, n_cp))

    for ts_idx in range(n_ts):
        ts = timeseries[:, ts_idx]
        comp = components[ts_idx, :, :]

        ts_cp_sses = np.apply_along_axis(
            lambda c: np.sum((ts - c) ** 2),
            0,
            comp
        )
        # Best component is one in which the residuals are least (contribute
        # the most to the reconstruction).
        optimal_order = np.argsort(ts_cp_sses)
        optimal_orders[:, ts_idx] = optimal_order

    return optimal_orders



def calculate_leverage(X):
    H = np.linalg.pinv(np.dot(X.T, X))
    H = np.linalg.multi_dot([X, H, X.T])
    return np.diag(H)


def calculate_aic(ssr, n=None, k=None):
    return n * np.log(ssr/n) + (2 * k)

def calculate_bic(ssr, n=None, k=None):
    return n * np.log(ssr/n) + (np.log(n) * k)


def batch_calculate_component_bics(N, P, cumulative_component_residuals):
    ssrs = np.sum(cumulative_component_residuals**2, axis=1)
    parameters = np.arange(1, ssrs.shape[1]+1)
    bics = np.apply_along_axis(calculate_bic, 1, ssrs, k=parameters, n=N)
    return bics


def batch_calculate_component_aics(N, P, cumulative_component_residuals):
    ssrs = np.sum(cumulative_component_residuals**2, axis=1)
    parameters = np.arange(1, ssrs.shape[1]+1)
    aics = np.apply_along_axis(calculate_aic, 1, ssrs, k=parameters, n=N)
    return aics



def _rdecomp_norm_p(X, p):
    X_ = np.power(X, p)
    return np.sum(X_)

def _rdecomp_shrink(X, tau):
    X_sign = np.sign(X)
    X_zero = np.zeros(X.shape)

    X1 = np.abs(X) - tau
    X2 = np.maximum(X1, X_zero)
    return (X_sign * X2)

def _rdecomp_svd_thresholded(X, tau, rank):
    #U, s, V = np.linalg.svd(X, full_matrices=False)
    U, s, V = randomized_svd(X, rank)
    rank = np.sum(s > 0)
    X1 = _rdecomp_shrink(s, tau)
    X2 = np.diag(X1)
    X3 = np.dot(X2, V)
    X4 = np.dot(U, X3)
    return X4, rank


def robust_svd(X, rank):
    # Code adapted from this:
    # https://github.com/dganguli/robust-pca/blob/master/r_pca.py
    S = np.zeros(X.shape)
    Y = np.zeros(X.shape)

    mu = np.prod(X.shape) / (4 * _rdecomp_norm_p(X, 2))
    mu_inv = 1 / mu
    lam = 1 / np.sqrt(np.max(X.shape))

    err = np.Inf
    tol = 1e-7 * _rdecomp_norm_p(np.abs(X), 2)

    i = 0
    while (err > tol) and (i < 50):
        L, rank = _rdecomp_svd_thresholded(X - S + mu_inv * Y, mu_inv, rank)
        S = _rdecomp_shrink(X - L + (mu_inv * Y), mu_inv * lam)
        Y = Y + mu * (X - L - S)
        err = _rdecomp_norm_p(np.abs(X - L - S), 2)
        i += 1

    #U, s, V = np.linalg.svd(S)
    U, s, V = randomized_svd(L, rank)
    return U, s, V



def ar_timeseries_simulation(timeseries, n_simulations=300, ar=1):
    n = timeseries.shape[0]
    ar_model = ARMA(timeseries, (ar, 0))
    ar_model_results = ar_model.fit()
    ar_process = ArmaProcess().from_estimation(ar_model_results)
    noise_std = np.sqrt(ar_model_results.sigma2)
    simulated = [
        ar_process.generate_sample(nsample=n, scale=noise_std, burnin=10)
        for i in range(n_simulations)
    ]
    simulated = np.array(simulated).T
    return simulated


def simulated_timeseries_eigenvalues(simulated_timeseries,
                                     eigenvectors,
                                     L,
                                     K):
    # http://research.atmos.ucla.edu/tcd//ssa/guide/andy/node1.html

    simulated_eigenvalues = []
    for i in range(simulated_timeseries.shape[1]):
        sts = simulated_timeseries[:, i]

        trajectory_matrix = ts_vector_to_trajectory_matrix(
            sts,
            L,
            K
        )

        Cs = np.cov(trajectory_matrix)

        sim_ev = eigenvectors.T @ Cs @ eigenvectors
        sim_ev = np.diag(sim_ev) ** 2
        simulated_eigenvalues.append(sim_ev)

    simulated_eigenvalues = np.array(simulated_eigenvalues)
    return simulated_eigenvalues


def monte_carlo_component_selection(timeseries,
                                    eigenvectors,
                                    eigenvalues,
                                    L,
                                    K,
                                    percentile_threshold=90):

    ar_simulations = ar_timeseries_simulation(timeseries)
    simulated_ev = simulated_timeseries_eigenvalues(
        ar_simulations,
        eigenvectors,
        L,
        K
    )

    thresholds = scipy.stats.scoreatpercentile(
        simulated_ev,
        percentile_threshold,
        axis=0
    )

    components = np.where(eigenvalues > thresholds)[0]
    return components



def monte_carlo_ssa_reconstruction(timeseries,
                                   L,
                                   K):
    trajectory_matrix = ts_vector_to_trajectory_matrix(
        timeseries,
        L,
        K
    )

    C = np.cov(trajectory_matrix)

    U, s, V = np.linalg.svd(C)
    eigenvalues = s ** 2

    components = monte_carlo_component_selection(
        timeseries,
        U,
        eigenvalues,
        L,
        K
    )

    elementary_matrix = np.zeros_like(trajectory_matrix)
    for c in components:
        reconstruction = elementary_matrix_at_rank_alt(
            trajectory_matrix,
            U,
            c
        )
        elementary_matrix += reconstruction

    reconstructed = np.zeros_like(timeseries)
    diagonal_averager(elementary_matrix, reconstructed)

    return reconstructed





#
# def varimax_rotation(U,
#                      max_iter=500,
#                      tol=1e-6):
#
#     r, c = U.shape
#     R = np.eye(c)
#
#     delta = 0
#     previous_delta = 0
#     for i in range(max_iter):
#         previous_delta = delta
#
#         B = np.dot(U, R)
#
#         T = np.dot(U.T, (B**3) - (1/r) * np.dot(B, np.diag(np.diag(np.dot(B.T, B)))))
#
#         U_, s_, V_ = np.linalg.svd(T)
#
#         R = np.dot(U_, V_)
#         delta = np.sum(s_)
#
#         if (previous_delta != 0) and ((delta / previous_delta) < (1 + tol)):
#             break
#
#     return R
#
#
# def promax_rotation(U,
#                     max_iter=500,
#                     tol=1e-6):
#
#     R = varimax_rotation(U)
#     X = U @ R
#
#     Y = X * np.abs(X)**3
#
#     beta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
#
#     diag_inv = np.diag(np.linalg.pinv(np.dot(beta.T, beta)))
#
#     beta = np.dot(beta, np.diag(np.sqrt(diag_inv)))
#
#     R = np.dot(R, beta)
#     return R

