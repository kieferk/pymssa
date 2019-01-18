import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=True)
def structured_varimax(U, n_timeseries, window, gamma=1, tol=1e-6, max_iter=2500):
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


@jit(nopython=True, fastmath=True, parallel=True)
def elementary_matrix_at_rank(trajectory_matrix,
                              left_singular_vectors,
                              rank):

    U_r = left_singular_vectors[:, rank:rank+1]
    X_r = np.dot(np.dot(U_r, U_r.T), trajectory_matrix)
    return X_r


@jit(nopython=True, fastmath=True, parallel=True)
def diagonal_averager(trajectory_matrix):
    # Reconstruct a timeseries from a trajectory matrix using diagonal
    # averaging procedure.
    # https://arxiv.org/pdf/1309.5050.pdf

    r_matrix = trajectory_matrix[::-1]
    unraveled = []
    for i in range(-r_matrix.shape[0]+1, r_matrix.shape[1]):
        tp_diag = np.diag(r_matrix, k=i)
        tp = np.mean(tp_diag)
        unraveled.append(tp)
    unraveled = np.array(unraveled)
    return unraveled


@jit(nopython=True, fastmath=True, parallel=True)
def diagonal_average_each_component(elementary_matrix,
                                    components):

    components = np.atleast_1d(components)
    reconstructions = []

    for c in components:
        at_component = elementary_matrix[:, :, c]
        recon = diagonal_averager(at_component)
        reconstructions.append(recon)

    reconstructions = np.array(reconstruction).T
    return reconstructions


@jit(nopython=True, fastmath=True, parallel=True)
def diagonal_average_at_components(elementary_matrix,
                                   components):
    reconstructions = diagonal_average_each_component(
        elementary_matrix,
        components
    )
    reconstruction = np.sum(reconstructions, axis=1)
    return reconstruction


@jit(nopython=True, fastmath=True, parallel=True)
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


@jit(nopython=True, fastmath=True, parallel=True)
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
            ts_component = ts_components[:, i]
            ts_comp_sq = ts_component * ts_component
            weighted_r = np.dot(weights, ts_comp_sq)
            norm_i = weighted_norms[i]
            norm_j = weighted_norms[j]
            corr = weighted_r * norm_i * norm_j
            weighted_correlation[i, j] = corr
            weighted_correlation[j, i] = corr

    return weighted_correlation


@jit(nopython=True, fastmath=True, parallel=True)
def construct_forecasting_matrix(n_timeseries,
                                 W_matrix,
                                 U_matrix):

    R1 = np.eye(n_timeseries)
    R2 = R1 - np.dot(W_matrix, W_matrix.T)
    R3 = np.linalg.pinv(R2)
    R4 = np.dot(R3, W_matrix)
    R5 = np.dot(R4, U_matrix.T)
    return R5


@jit(nopython=True, fastmath=True, parallel=True)
def optimal_component_ordering(timeseries,
                               components):

    optimal_orders = np.zeros((components.shape[2], components.shape[0]))

    for ts_idx in range(timeseries.shape[1]):
        ts = timeseries[:, ts_idx:(ts_idx+1)]
        comp = components[ts_idx, :, :]

        residuals = ts - comp

        maes = []
        for ridx in range(residuals.shape[1]):
            resid = residuals[:, ridx]
            mae = np.mean(np.abs(resid))
            maes.append(mae)

        maes = np.array(maes)
        optimal_order = np.argsort(maes)
        optimal_orders[:, ts_idx] = optimal_order

    optimal_orders = optimal_orders.astype(int)
    return optimal_orders
