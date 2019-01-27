import numpy as np
from numba import jit

from .ops import *


@jit(nopython=True, fastmath=True)
def structured_varimax(U, n_timeseries, window, gamma=1, tol=1e-8, max_iter=5000):
    # See:
    # http://200.145.112.249/webcast/files/SeminarMAR2017-ICTP-SAIFR.pdf

    # get the shape of the singular vectors
    p, k = U.shape

    # initialize the varimax rotation to identity matrix
    T = np.eye(k)

    # initialize singular value sum tracker
    d = 0

    # rename to match variable names in code referenced in paper above
    # for clarity
    D = n_timeseries
    M = window

    # initialize matrices
    vec_i = np.ones(M).reshape((1, M))
    I_d = np.eye(D)

    # kronecker product
    I_d_md = np.kron(I_d, vec_i)

    M = I_d - (gamma / D) * np.ones((D, D))
    IMI = np.dot(I_d_md.T, np.dot(M, I_d_md))

    d_old = 0
    iteration = 0

    while (d_old == 0) or ((d / d_old) > (1 + tol)):

        d_old = d
        iteration = iteration + 1

        B = np.dot(U, T)
        G = np.dot(U.T, (B * np.dot(IMI, B ** 2)))

        u, s, vh = np.linalg.svd(G)

        T = np.dot(u, vh)
        d = np.sum(s)

        if iteration >= max_iter:
            break

    return T



@jit(nopython=True, fastmath=True)
def calculate_factor_vectors(trajectory_matrix,
                             left_singular_vectors,
                             singular_values,
                             rank):
    # The "factor vectors" are defined as X.T U / sqrt(s)
    # Where X is the trajectory matrix, U is the left-singular vectors,
    # and s are the singular values
    U = left_singular_vectors[:, :rank]
    factor_vectors = np.dot(trajectory_matrix.T, U) / singular_values[:rank]
    factor_vectors = factor_vectors.T
    return factor_vectors


@jit(nopython=True, fastmath=True)
def elementary_matrix_at_rank(trajectory_matrix,
                              left_singular_vectors,
                              rank):

    U_r = left_singular_vectors[:, rank:rank+1]
    X_r = np.dot(np.dot(U_r, U_r.T), trajectory_matrix)
    return X_r


@jit(nopython=True, fastmath=True)
def construct_elementary_matrix(trajectory_matrix,
                                left_singular_vectors,
                                singular_values,
                                rank):
    # Elementary matrices are reconstructions of the trajectory matrices
    # from a set of left singular vectors and singular values

    elementary_matrix = np.zeros((trajectory_matrix.shape[0],
                                  trajectory_matrix.shape[1],
                                  rank))

    for r in range(rank):
        elementary_matrix[:, :, r] = elementary_matrix_at_rank(
            trajectory_matrix,
            left_singular_vectors,
            r
        )

    return elementary_matrix


@jit(nopython=True, fastmath=True)
def vtmat_ts_startidx(timeseries_index, L):
    return L * timeseries_index



@jit(nopython=True, fastmath=True)
def vtmat_ts_endidx(timeseries_index, L):
    return L * (timeseries_index + 1)


@jit(nopython=True, fastmath=True)
def elementary_matrix_for_timeseries_index(elementary_matrix, ts_idx, L):
    sidx = vtmat_ts_startidx(ts_idx, L)
    eidx = vtmat_ts_endidx(ts_idx, L)
    return elementary_matrix[sidx:eidx]


@jit(nopython=True, fastmath=True)
def reshape_elementary_matrix_by_timeseries_index(elementary_matrix, P, L):
    _, K, rank = elementary_matrix.shape
    elementary_matrices = np.zeros((P, L, K, rank))

    for ts_idx in range(P):
        elementary_matrices[ts_idx, :, :, :] = elementary_matrix_for_timeseries_index(
            elementary_matrix,
            ts_idx,
            L
        )

    return elementary_matrices


@jit(nopython=True, fastmath=True)
def diagonal_averager(elementary_matrix, allocated_output):
    # Reconstruct a timeseries from a trajectory matrix using diagonal
    # averaging procedure.
    # https://arxiv.org/pdf/1309.5050.pdf

    r_matrix = elementary_matrix[::-1]
    for i, d in enumerate(range(-r_matrix.shape[0]+1, r_matrix.shape[1])):
        tp_diag = np.diag(r_matrix, k=d)
        allocated_output[i] = np.mean(tp_diag)



@jit(nopython=True, fastmath=True)
def batch_diagonal_averager(elementary_matrix, component_matrix_ref):
    em_rev = elementary_matrix[:, ::-1, :]
    lidx, ridx = -em_rev.shape[1]+1, em_rev.shape[2]
    for i, offset in enumerate(range(lidx, ridx)):
        for t in range(em_rev.shape[0]):
            diag = np.diag(em_rev[t], k=offset)
            component_matrix_ref[t, i] = np.mean(diag)



@jit(nopython=True, fastmath=True)
def diagonal_average_each_component(elementary_matrix,
                                    components,
                                    N):

    components = np.atleast_1d(components)
    reconstructions = np.zeros((N, len(components)))

    for i, c in enumerate(components):
        at_component = elementary_matrix[:, :, c]
        diagonal_averager(at_component, reconstructions[:, i])

    return reconstructions



@jit(nopython=True, fastmath=True)
def diagonal_average_at_components(elementary_matrix,
                                   components,
                                   N):
    reconstructions = diagonal_average_each_component(
        elementary_matrix,
        components,
        N
    )
    reconstruction = np.sum(reconstructions, axis=1)
    return reconstruction





@jit(nopython=True, fastmath=True)
def _incremental_component_reconstruction_inner(trajectory_matrix,
                                                components,
                                                left_singular_vectors,
                                                P,
                                                L):

    for r in range(components.shape[2]):
        elementary_matrix_r = elementary_matrix_at_rank(
            trajectory_matrix,
            left_singular_vectors,
            r
        )

        batch_diagonal_averager(
            elementary_matrix_r.reshape((P, L, -1)),
            components[:, :, r]
        )

    return components



@jit(nopython=True, fastmath=True)
def incremental_component_reconstruction(trajectory_matrix,
                                      left_singular_vectors,
                                      singular_values,
                                      rank,
                                      P,
                                      N,
                                      L):

    components = np.zeros((P, N, rank))

    components = _incremental_component_reconstruction_inner(
        trajectory_matrix,
        components,
        left_singular_vectors,
        P,
        L
    )

    return components



@jit(nopython=True, fastmath=True)
def construct_hankel_weights(L, K, N):
    L_star = np.minimum(L, K)
    K_star = np.maximum(L, K)

    weights = []
    for i in range(N):
        if i <= (L_star - 1):
            weights.append(i+1)
        elif i <= K_star:
            weights.append(L_star)
        else:
            weights.append(N - i)

    weights = np.array(weights)
    return weights


@jit(nopython=True, fastmath=True)
def hankel_weighted_correlation(ts_components,
                                weights):

    weighted_norms = []
    for i in range(ts_components.shape[1]):
        ts_component_sq = ts_components[:, i] ** 2
        norm = np.dot(weights, ts_component_sq)
        weighted_norms.append(norm)

    weighted_norms = np.array(weighted_norms)
    weighted_norms = weighted_norms ** -0.5

    weighted_correlation = np.identity(ts_components.shape[1])
    M = weighted_correlation.shape[1]
    for i in range(M):
        for j in range(i+1, M):
            ts_comp_sq = ts_components[:, i] * ts_components[:, j]
            weighted_r = np.dot(weights, ts_comp_sq)
            corr = weighted_r * weighted_norms[i] * weighted_norms[j]
            weighted_correlation[i, j] = corr
            weighted_correlation[j, i] = corr

    return weighted_correlation



@jit(nopython=True, fastmath=True)
def optimal_component_ordering(timeseries,
                               components):

    optimal_orders = np.zeros((components.shape[2], components.shape[0]))

    for ts_idx in range(timeseries.shape[1]):
        ts = timeseries[:, ts_idx:(ts_idx+1)]
        comp = components[ts_idx, :, :]

        residuals = ts - comp

        maes = np.zeros(residuals.shape[1])
        for ridx in range(residuals.shape[1]):
            resid = residuals[:, ridx]
            maes[ridx] = np.mean(np.abs(resid))

        optimal_order = np.argsort(maes)
        optimal_orders[:, ts_idx] = optimal_order

    return optimal_orders



@jit(nopython=True, fastmath=True)
def construct_forecasting_matrix(n_timeseries,
                                 W_matrix,
                                 U_matrix):

    R1 = np.eye(n_timeseries)
    R2 = R1 - np.dot(W_matrix, W_matrix.T)
    R3 = np.linalg.pinv(R2)
    R4 = np.dot(R3, W_matrix)
    R5 = np.dot(R4, U_matrix.T)
    return R5



@jit(nopython=True, fastmath=True)
def forecasting_matrix_for_components(left_singular_vectors,
                                      components,
                                      P,
                                      L):

    components = np.atleast_1d(components)

    U = np.zeros((L * P - P, len(components)))
    W = np.zeros((P, len(components)))

    for p in range(P):
        sidx = vtmat_ts_startidx(p, L)
        eidx = vtmat_ts_endidx(p, L)
        ts_lsv = left_singular_vectors[sidx:(eidx - 1), :]
        ts_lsv = ts_lsv[:, components]

        u_sidx = sidx - p
        u_eidx = u_sidx + (L - 1)
        U[u_sidx:u_eidx, :] = ts_lsv

        w = left_singular_vectors[(eidx - 1), :]
        w = w[components]
        W[p, :] = w

    R = construct_forecasting_matrix(P, W, U)
    return R



@jit(nopython=True, fastmath=True)
def vmssa_recurrent_forecast(timepoints_out,
                             components,
                             left_singular_vectors,
                             P,
                             L,
                             use_components):

    recons = components[:, :, use_components]
    recons = recons.sum(axis=2)

    R = forecasting_matrix_for_components(
        left_singular_vectors,
        use_components,
        P,
        L
    )

    Z = np.zeros((P, (L - 1) + timepoints_out))
    Z[:, :(L - 1)] = recons[:, (-L + 1):]

    Z_flat = np.zeros((P * (L - 1), 1))

    for t in range(timepoints_out):
        Z_ = Z[:, t:(L - 1 + t)]
        Z_flat[:, 0] = Z_.ravel()

        forecast_ = np.dot(R, Z_flat)

        Z[:, (L - 1 + t)] = forecast_.ravel()

    forecasted = Z[:, -timepoints_out:]

    return forecasted
