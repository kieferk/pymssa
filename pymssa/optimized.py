import numpy as np
from numba import jit, prange

#from .ops import *


#@jit(nopython=True, fastmath=True)
def structured_varimax(U, n_timeseries, window, gamma=1, tol=1e-8, max_iter=5000):
    # See:
    # http://200.145.112.249/webcast/files/SeminarMAR2017-ICTP-SAIFR.pdf

    #print('U shape:', U.shape)
    # get the shape of the singular vectors
    p, k = U.shape

    # initialize the varimax rotation to identity matrix
    T = np.eye(k)
    #print('T:', T.shape)

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
    #print('Idmd:', I_d_md.shape)

    M = I_d - (gamma / D) * np.ones((D, D))
    #print('M:', M.shape)
    IMI = np.dot(I_d_md.T, np.dot(M, I_d_md))
    #print('IMI:', IMI.shape)

    d_old = 0
    iteration = 0

    while (d_old == 0) or ((d / d_old) > (1 + tol)):

        d_old = d
        iteration = iteration + 1

        B = np.dot(U, T)
        #print('B:', B.shape)
        G = np.dot(U.T, (B * np.dot(IMI, B ** 2)))

        u, s, vh = np.linalg.svd(G)

        T = np.dot(u, vh)
        d = np.sum(s)

        if iteration >= max_iter:
            break

    return T


# @jit(nopython=True, fastmath=True)
# def _rdecomp_norm_p(X, p):
#     X_ = np.power(X, p)
#     return np.sum(X_)
#
# @jit(nopython=True, fastmath=True)
# def _rdecomp_shrink(X, tau):
#     X_sign = np.sign(X)
#     X_zero = np.zeros(X.shape)
#
#     X1 = np.abs(X) - tau
#     X2 = np.maximum(X1, X_zero)
#     return (X_sign * X2)
#
# @jit(nopython=True, fastmath=True)
# def _rdecomp_svd_thresholded(X, tau):
#     U, s, V = np.linalg.svd(X, full_matrices=False)
#     X1 = _recomp_shrink(s, tau)
#     X2 = np.diag(X1)
#     X3 = np.dot(X2, V)
#     X4 = np.dot(U, X3)
#     return X4
#
#
# @jit(nopython=True, fastmath=True)
# def robust_svd(X):
#     S = np.zeros(X.shape)
#     Y = np.zeros(X.shape)
#
#     mu = (X.shape[0] * X.shape[1]) / (4 * _rdecomp_norm_p(X, 2))
#     mu_inv = 1 / mu
#     lam = 1 / np.sqrt(np.max(X.shape[0], X.shape[1]))
#
#     err = np.Inf
#     tol = 1e-7 * _rdecomp_norm_p(np.abs(X), 2)
#
#     i = 0
#     while (err > tol) and (i < 1000):
#         L = _rdecomp_svd_thresholded(X - S + mu_inv * Y, mu_inv)
#         S = _rdecomp_shrink(X - L + (mu_inv * Y), mu_inv * lam)
#         Y = Y + mu * (X - L - S)
#         err = _rdecomp_norm_p(np.abs(X - L - S), 2)
#         i += 1
#
#     U, s, V = np.linalg.svd(L)
#     return U, s, V



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
def elementary_matrix_at_rank_alt(trajectory_matrix,
                                  left_singular_vectors,
                                  rank):

    U_r = left_singular_vectors[:, rank:rank+1]
    X_r = np.dot(np.dot(U_r, U_r.T), trajectory_matrix)
    return X_r


@jit(nopython=True, fastmath=True)
def elementary_matrix_at_rank(left_singular_vectors,
                              factor_vectors,
                              singular_values,
                              rank):

    U_r = left_singular_vectors[:, rank:rank+1]
    V_r = factor_vectors[rank:rank+1, :]
    X_r = np.dot(U_r, V_r) * singular_values[rank]
    return X_r


@jit(nopython=True, fastmath=True)
def construct_elementary_matrix(trajectory_matrix,
                                left_singular_vectors,
                                factor_vectors,
                                singular_values,
                                rank):
    # Elementary matrices are reconstructions of the trajectory matrices
    # from a set of left singular vectors and singular values

    elementary_matrix = np.zeros((trajectory_matrix.shape[0],
                                  trajectory_matrix.shape[1],
                                  rank))

    for r in range(rank):
        # elementary_matrix[:, :, r] = elementary_matrix_at_rank(
        #     trajectory_matrix,
        #     left_singular_vectors,
        #     factor_vectors,
        #     singular_values,
        #     r
        # )

        elementary_matrix[:, :, r] = elementary_matrix_at_rank_alt(
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
def batch_diagonal_averager(elementary_matrix,
                            component_matrix_ref):

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
def batch_dehankelization_inserter(dehankelization_matrix,
                                   elementary_matrix,
                                   row_indexer,
                                   col_indexer,
                                   P,
                                   R):

    for r in range(R):
        EM_r = elementary_matrix[:, :, :, r].reshape((P, -1))
        dehankelization_matrix[:, row_indexer, col_indexer, r] = EM_r

    return dehankelization_matrix



@jit(nopython=True, fastmath=True)
def reshape_elementary_matrix_by_timeseries(elementary_matrix,
                                            P,
                                            L,
                                            K):
    em_N, em_M = elementary_matrix.shape

    if em_N == (P * L):
        elementary_matrix_rs = elementary_matrix.reshape((P, L, -1))

    elif em_N == L:
        elementary_matrix_rs = np.zeros((P, L, K))
        for p in range(P):
            elementary_matrix_rs[p, :, :] = elementary_matrix[:, (p * K):((p + 1) * K)]

    return elementary_matrix_rs



@jit(nopython=True, fastmath=True)
def incremental_component_reconstruction_inner(components,
                                                left_singular_vectors,
                                                factor_vectors,
                                                singular_values,
                                                P,
                                                L,
                                                K):

    for r in range(components.shape[-1]):
        elementary_matrix_r = elementary_matrix_at_rank(
            left_singular_vectors,
            factor_vectors,
            singular_values,
            r
        )

        elementary_matrix_rs = reshape_elementary_matrix_by_timeseries(
            elementary_matrix_r,
            P,
            L,
            K
        )

        # for p in range(P):
        #     em_p_values = elementary_matrix_rs[p, :, :].ravel()
        #     for idx, (ri, ci) in enumerate(zip(row_indexer, col_indexer)):
        #         components[p, ri, ci, r] = em_p_values[idx]

        batch_diagonal_averager(
            elementary_matrix_rs,
            components[:, :, r],
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
def calculate_loocv(residuals, leverage):
    return np.mean((residuals / (1 - leverage)) ** 2)


@jit(nopython=True, fastmath=True)
def order_components(components,
                     component_orders):

    ordered_components = np.zeros_like(components)
    for ts_idx in range(components.shape[0]):
        ts_components = components[ts_idx, :, :]
        order = component_orders[:, ts_idx]
        ordered_components[ts_idx, :, :] = ts_components[:, order]

    return ordered_components


@jit(nopython=True, fastmath=True)
def ordered_cumulative_component_residuals(timeseries,
                                           cumulative_ordered_components):

    residuals = np.zeros_like(cumulative_ordered_components)

    for ts_idx in range(timeseries.shape[1]):
        ts = timeseries[:, ts_idx:(ts_idx + 1)]
        ts_components = cumulative_ordered_components[ts_idx, :, :]
        residuals[ts_idx, :, :] = ts - ts_components

    return residuals




@jit(nopython=True, fastmath=True)
def batch_calculate_component_loocvs(N, P,
                                     components,
                                     cumulative_component_residuals,
                                     leverages):

    loocvs = np.zeros((components.shape[2], P))
    loocvs[:] = np.nan

    for ts_idx in range(P):
        residuals = cumulative_component_residuals[ts_idx, :, :]

        for component_idx in range(components.shape[2]):
            leverage = leverages[ts_idx, :, component_idx]
            loocvs[component_idx, ts_idx] = calculate_loocv(
                residuals[:, component_idx],
                leverage
            )

    return loocvs


@jit(nopython=True, fastmath=True)
def construct_vmssa_forecasting_matrix(n_timeseries,
                                       W_matrix,
                                       U_matrix):

    R1 = np.eye(n_timeseries)
    R2 = R1 - np.dot(W_matrix, W_matrix.T)
    R3 = np.linalg.pinv(R2)
    R4 = np.dot(R3, W_matrix)
    R5 = np.dot(R4, U_matrix.T)
    return R5



@jit(nopython=True, fastmath=True)
def vmssa_forecasting_matrix_for_components(left_singular_vectors,
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

    R = construct_vmssa_forecasting_matrix(P, W, U)
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

    R = vmssa_forecasting_matrix_for_components(
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


def hmssa_forecasting_matrix_for_components(left_singular_vectors,
                                            components,
                                            L):

    components = np.atleast_1d(components)

    U = np.zeros((L - 1, len(components)))
    W = np.zeros(len(components))

    U[:, :] = left_singular_vectors[:-1, components]
    W[:] = left_singular_vectors[-1, components]

    v_sq = np.sum(W ** 2)

    R = np.zeros((1, U.shape[0]))
    R[0, :] = (1 / (1 - v_sq)) * np.sum(W * U, axis=1)
    return R


#@jit(nopython=True, fastmath=True)
def hmssa_recurrent_forecast(timepoints_out,
                             components,
                             left_singular_vectors,
                             P,
                             L,
                             use_components):

    recons = components[:, :, use_components]
    recons = recons.sum(axis=2)

    R = hmssa_forecasting_matrix_for_components(
        left_singular_vectors,
        use_components,
        L
    )

    Z = np.zeros((P, (L - 1) + timepoints_out))
    Z[:, :(L - 1)] = recons[:, (-L + 1):]

    for t in range(timepoints_out):
        Z_ = Z[:, t:(L - 1 + t)]
        forecast_ = np.dot(R, Z_.T)
        Z[:, (L - 1 + t)] = forecast_.ravel()

    forecasted = Z[:, -timepoints_out:]

    return forecasted

