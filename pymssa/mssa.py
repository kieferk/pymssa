import numpy as np
from numpy.linalg import matrix_rank

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import hankel

from functools import partial, lru_cache, reduce
from tqdm.autonotebook import tqdm

from .optimized import (
    structured_varimax,
    elementary_matrix_at_rank,
    diagonal_averager,
    diagonal_average_each_component,
    diagonal_average_at_components,
    hankel_weighted_correlation,
    construct_forecasting_matrix,
    optimal_component_ordering,
    construct_hankel_weights
)

from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import explained_variance_score



class MSSA:

    def __init__(self,
                 window_size=None,
                 n_components=None,
                 variance_explained_threshold=0.95,
                 pa_percentile_threshold=95,
                 svd_method='randomized',
                 varimax=False,
                 verbose=True):

        self.set_params(window_size=window_size,
                        n_components=n_components,
                        variance_explained_threshold=variance_explained_threshold,
                        pa_percentile_threshold=pa_percentile_threshold,
                        svd_method=svd_method,
                        varimax=varimax,
                        verbose=verbose)


    def get_params(self,
                   deep=True):
        '''get_params method for compliance with sklearn model api.'''
        return dict(
            window_size=self.window_size,
            n_components=self.n_components,
            variance_explained_threshold=self.variance_explained_threshold,
            pa_percentile_threshold=self.pa_percentile_threshold,
            svd_method=self.svd_method,
            varimax=self.varimax,
            verbose=self.verbose
        )


    def set_params(self,
                   **parameters):
        '''set_params method for compliance with sklearn model api.'''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def hankelize_timeseries(self,
                             timeseries,
                             L,
                             K):
        hankelized = hankel(timeseries, np.zeros(L)).T
        hankelized = hankelized[:, :K]
        return hankelized


    def create_trajectory_matrix(self,
                                 timeseries,
                                 L,
                                 K):
        hankelized = [
            self.hankelize_timeseries(timeseries[:, p], L, K)
            for p in range(timeseries.shape[1])
            ]

        # V-MSSA formulation
        # https://www.researchgate.net/publication/263870252_Multivariate_singular_spectrum_analysis_A_general_view_and_new_vector_forecasting_approach
        trajectory_matrix = np.concatenate(hankelized, axis=0)
        return trajectory_matrix



    def decompose_trajectory_matrix(self,
                                    trajectory_matrix,
                                    K):

        # calculate S matrix
        # https://arxiv.org/pdf/1309.5050.pdf
        S = np.dot(trajectory_matrix, trajectory_matrix.T)

        # Perform SVD on S
        if self.svd_method == 'randomized':
            U, s, V = randomized_svd(S, K)
        elif self.svd_method == 'exact':
            U, s, V = np.linalg.svd(S)

        # Valid rank is only where eigenvalues > 0
        rank = np.sum(s > 0)

        # singular values are the square root of the eigenvalues
        s = np.sqrt(s)

        return U, s, V, rank


    def apply_structured_varimax(self,
                                 left_singular_vectors,
                                 singular_values,
                                 P,
                                 L,
                                 gamma=1,
                                 tol=1e-6,
                                 max_iter=1000):
        # http://200.145.112.249/webcast/files/SeminarMAR2017-ICTP-SAIFR.pdf

        T = structured_varimax(
            left_singular_vectors,
            P,
            L,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter
        )

        U = left_singular_vectors @ T
        slen = singular_values.shape[0]
        s = np.diag(T[:slen, :slen].T @ np.diag(singular_values) @ T[:slen, :slen])

        return U, s


    def calculate_explained_variance_ratio(self,
                                           singular_values):
        # Calculation taken from sklearn. See:
        # https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/decomposition/pca.py
        eigenvalues = singular_values ** 2
        explained_variance = eigenvalues / (self.N_ - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        return explained_variance, explained_variance_ratio



    def adjust_rank_with_svht(self,
                              singular_values,
                              rank):
        # Singular Value Hard Thresholding
        # This is a threshold on the rank/singular values based on the findings
        # in this paper:
        # https://arxiv.org/pdf/1305.5870.pdf
        # We assume the noise is not known, and so the thresholding value is
        # determined by the data (See section D)
        median_sv = np.median(singular_values[:rank])
        threshold = 2.858 * median_sv
        adjusted_rank = np.sum(singular_values >= threshold)
        return adjusted_rank


    def parallel_analysis_component_selection(self,
                                              timeseries,
                                              L,
                                              K,
                                              rank,
                                              singular_values,
                                              iterations=100):

        def _bootstrap_eigenvalues(ts_std, ts_shape, L, K, rank):

            # create random normal differences with equivalent standard deviations
            ts_rnorm = np.random.normal(
                np.zeros(ts_shape[1]),
                ts_std,
                size=ts_shape
            )

            # create noise trajectory matrix
            rnorm_trajectory_matrix = self.create_trajectory_matrix(
                ts_rnorm,
                L,
                K
            )

            # decompose the noise trajectory matrix
            U, s, V, rank = self.decompose_trajectory_matrix(
                rnorm_trajectory_matrix,
                rank
            )

            # return the eigenvalues
            return s ** 2

        # calculate real eigenvalues
        eigenvalues = singular_values ** 2

        # calculate standard deviations column-wise
        ts_std = np.std(timeseries, axis=0)

        # bootstrap the eigenvalues
        noise_eigenvalues = [
            _bootstrap_eigenvalues(
                ts_std,
                timeseries.shape,
                L,
                K,
                rank
            )
            for i in tqdm(range(iterations), disable=(not self.verbose))
        ]
        noise_eigenvalues = np.concatenate(noise_eigenvalues, axis=0)

        # calculate the 95th percentile of the noise eigenvalues
        eig_pctl = np.percentile(noise_eigenvalues, 95, axis=0)

        # find the first index where the noise eigenvalue 95th percentile is >= real
        adjusted_rank = np.where(eig_pctl > eigenvalues)[0][0]

        return adjusted_rank



    def create_factor_vectors(self,
                              trajectory_matrix,
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


    def _get_lidx(self, timeseries_index):
        # Finds the appropriate start index for a specific timeseries
        return self.L_ * timeseries_index

    def _get_ridx(self, timeseries_index):
        # Finds the appropriate end index for a specific timeseries
        return self.L_ * (timeseries_index + 1)


    def create_elementary_matrices(self,
                                   trajectory_matrix,
                                   left_singular_vectors,
                                   singular_values,
                                   rank):
        # Elementary matrices are reconstructions of the trajectory matrices
        # from a set of left singular vectors and singular values

        elementary_matrix = np.zeros((trajectory_matrix.shape[0],
                                      trajectory_matrix.shape[1],
                                      rank))

        for r in tqdm(range(rank), disable=(not self.verbose)):
            # From section VMSSA-R
            elementary_matrix[:, :, r] = elementary_matrix_at_rank(
                trajectory_matrix,
                left_singular_vectors,
                r
            )

        elementary_matrices = np.zeros((self.P_, self.L_, self.K_, rank))

        for ts_idx in range(self.P_):
            lidx = self._get_lidx(ts_idx)
            ridx = self._get_ridx(ts_idx)
            elementary_matrices[ts_idx, :, :, :] = elementary_matrix[lidx:ridx, :, :]

        return elementary_matrices



    # def diagonal_averaging(self,
    #                        trajectory_matrix):
    #     # Reconstruct a timeseries from a trajectory matrix using diagonal
    #     # averaging procedure.
    #     # https://arxiv.org/pdf/1309.5050.pdf
    #
    #     r_matrix = trajectory_matrix[::-1]
    #     unraveled = [
    #         r_matrix.diagonal(i).mean()
    #         for i in range(-r_matrix.shape[0]+1, r_matrix.shape[1])
    #     ]
    #     return np.array(unraveled)
    #
    #
    # def reconstruct_timeseries(self,
    #                            elementary_matrices,
    #                            timeseries_index,
    #                            components):
    #
    #     components = np.atleast_1d(components)
    #     ts_elementary_matrix = self.elementary_matrices_[timeseries_index, :, :, :]
    #
    #     reconstruction = [
    #         self.diagonal_averaging(ts_elementary_matrix[:, :, c])
    #         for c in components
    #     ]
    #
    #     reconstruction = np.array(reconstruction).sum(axis=0)
    #     return reconstruction


    def calculate_optimal_reconstruction_orders(self,
                                                timeseries,
                                                components):

        # optimal_orders = np.zeros((components.shape[2], components.shape[0]))
        # for ts_idx in range(timeseries.shape[1]):
        #     residuals = timeseries[:, ts_idx:(ts_idx+1)] - components[ts_idx, :, :]
        #     maes = np.apply_along_axis(lambda r: np.mean(np.abs(r)), 0, residuals)
        #     optimal_order = np.argsort(maes)
        #     optimal_orders[:, ts_idx] = optimal_order
        #
        # optimal_orders = optimal_orders.astype(int)
        optimal_orders = optimal_component_ordering(
            timeseries,
            components
        )

        order_explained_variance = np.zeros_like(optimal_orders).astype(float)
        for ts_idx in range(timeseries.shape[1]):
            ts_comp = components[ts_idx, :, :]
            ts_comp = ts_comp[:, optimal_orders[:, ts_idx]]
            ts_comp = np.cumsum(ts_comp, axis=1)

            order_explained_variance[:, ts_idx] = np.apply_along_axis(
                partial(explained_variance_score, timeseries[:, ts_idx]),
                0,
                ts_comp
            )

        return optimal_orders, order_explained_variance



    def _validate_initialization_arguments(self):
        # Check the window size parameter
        if self.window_size is not None:
            if not isinstance(self.window_size, int):
                raise Exception("window_size must be an integer (or None).")
            if self.window_size > (self.N_ // 2):
                raise Exception("window_size must be <= (timeseries length // 2).")

        # Check the components parameter
        if self.n_components is not None:
            if isinstance(self.n_components, str):
                comp_options = ['variance_threshold','svht','parallel_analysis']
                if self.n_components not in comp_options:
                    raise Exception('automatic n_component selections mus be one of:', comp_options)
            elif isinstance(self.n_components, int):
                if self.n_components > (self.N_ - self.L_ + 1):
                    raise Exception("Too many n_components specified for given window_size.")
                if self.n_components < 1:
                    raise Exception("n_components cannot be set < 1.")
            else:
                raise Exception('Invalid value for n_components set.')

        # Check variance explained threshold
        if self.variance_explained_threshold is not None:
            if not (self.variance_explained_threshold > 0):
                raise Exception("variance_explained_threshold must be > 0 (or None).")
            if not (self.variance_explained_threshold <= 1):
                raise Exception("variance_explained_threshold must be <= 1 (or None).")
        elif self.n_components == 'variance_threshold':
            raise Exception("If n_components == 'variance_threshold', variance_explained_threshold cannot be None.")

        # check parallel analysis threshold
        if self.pa_percentile_threshold is None and self.n_components == 'auto':
            raise Exception("If n_components == 'auto', pa_percentile_threshold must be specified.")
        if self.pa_percentile_threshold is not None:
            if (self.pa_percentile_threshold <= 0) or (self.pa_percentile_threshold > 100):
                raise Exception("pa_percentile_threshold must be > 0 and <= 100.")

        # check svd method
        if not self.svd_method in ['randomized', 'exact']:
            raise Exception("svd_method must be one of 'randomized', 'exact'.")



    def fit(self,
            timeseries):

        if timeseries.ndim == 1:
            timeseries = timeseries[:, np.newaxis]

        self.timeseries_ = timeseries
        self.N_ = timeseries.shape[0]
        self.P_ = timeseries.shape[1]
        self.L_ = (self.N_ // 2)

        self._validate_initialization_arguments()

        if self.window_size is not None:
            self.L_ = self.window_size

        self.K_ = self.N_ - self.L_ + 1

        if self.verbose:
            print('Constructing trajectory matrix')

        self.trajectory_matrix_ = self.create_trajectory_matrix(
            self.timeseries_,
            self.L_,
            self.K_
        )

        if self.verbose:
            print('Trajectory matrix shape:', self.trajectory_matrix_.shape)

        if self.verbose:
            print('Decomposing trajectory covariance matrix with SVD')

        U, s, V, rank = self.decompose_trajectory_matrix(
            self.trajectory_matrix_,
            self.K_
        )
        self.rank_ = rank
        self.left_singular_vectors_ = U
        self.singular_values_ = s

        if self.varimax:
            if self.verbose:
                print('Applying structured varimax to singular vectors')

            self.left_singular_vectors_, self.singular_values_ = self.apply_structured_varimax(
                self.left_singular_vectors_,
                self.singular_values_,
                self.P_,
                self.L_
            )

        exp_var, exp_var_ratio = self.calculate_explained_variance_ratio(
            self.singular_values_
        )
        self.explained_variance_ = exp_var
        self.explained_variance_ratio_ = exp_var_ratio

        if self.n_components == 'svht':
            self.rank_ = self.adjust_rank_with_svht(
                self.singular_values_,
                self.rank_
            )
            if self.verbose:
                print('Reduced rank to {} according to SVHT threshold'.format(self.rank_))

        elif self.n_components == 'variance_threshold':
            exp_var_ratio_cs = np.cumsum(exp_var_ratio)
            cutoff_n = np.sum(exp_var_ratio_cs <= self.variance_explained_threshold)
            self.rank_ = cutoff_n

            if self.verbose:
                print('Reduced rank to {} according to variance explained threshold'.format(self.rank_))

        elif self.n_components == 'parallel_analysis':
            if self.verbose:
                print('Performing parallel analysis to determine optimal rank')

            self.rank_ = self.parallel_analysis_component_selection(
                self.timeseries_,
                self.L_,
                self.K_,
                self.rank_,
                self.singular_values_
            )

            if self.verbose:
                print('Rank selected via parallel analysis: {}'.format(self.rank_))

        elif isinstance(self.n_components, int):
            self.rank_ = np.minimum(self.rank_, self.n_components)

        if self.verbose:
            print('Constructing factor vectors')

        self.factor_vectors_ = self.create_factor_vectors(
            self.trajectory_matrix_,
            self.left_singular_vectors_,
            self.singular_values_,
            self.rank_
        )

        if self.verbose:
            print('Constructing elementary matrices')

        self.elementary_matrices_ = self.create_elementary_matrices(
            self.trajectory_matrix_,
            self.left_singular_vectors_,
            self.singular_values_,
            self.rank_
        )

        if self.verbose:
            print('Constructing components')

        self.components_ = np.zeros((self.P_, self.N_, self.rank_))
        for ts_idx in tqdm(range(self.P_), disable=(not self.verbose)):
            ts_matrix = self.elementary_matrices_[ts_idx, :, :, :]
            components = np.arange(self.rank_)
            self.components_[ts_idx, :, :] = diagonal_average_each_component(
                ts_matrix,
                components
            )

        if self.verbose:
            print('Calculating optimal reconstruction orders')

        ranks, rank_exp_var = self.calculate_optimal_reconstruction_orders(
            self.timeseries_,
            self.components_
        )
        self.component_ranks_ = ranks
        self.component_ranks_explained_variance_ = rank_exp_var



    @property
    def hankel_weights_(self):
        # L_star = np.minimum(self.L_, self.K_)
        # K_star = np.maximum(self.L_, self.K_)
        #
        # # special weights due to the the hankelized trajectory matrix
        # weights = [i+1 if i <= (L_star - 1) else
        #            L_star if i <= (K_star) else
        #            self.N_ - i
        #            for i in range(self.N_)]
        # weights = np.array(weights)
        weights = construct_hankel_weights(
            self.L_,
            self.K_,
            self.N_
        )

        return weights


    # def w_correlation(self, ts_components):
    #     weights = self.hankel_weights_
    #     w_norms = np.array([np.dot(weights, ts_components[:, i]**2)
    #                         for i in range(ts_components.shape[1])])
    #     w_norms = w_norms ** -0.5
    #
    #     w_corr = np.identity(ts_components.shape[1])
    #     for i in range(w_corr.shape[0]):
    #         for j in range(i+1, w_corr.shape[0]):
    #             r = np.dot(weights, ts_components[:, i] * ts_components[:, j])
    #             r = np.abs(r * w_norms[i] * w_norms[j])
    #             w_corr[i, j] = r
    #             w_corr[j, i] = r
    #
    #     return w_corr


    def w_correlation(self, ts_components):
        weights = self.hankel_weights_
        w_corr = hankel_weighted_correlation(
            ts_components,
            weights
        )
        return w_corr


    def abs_w_correlation(self, ts_components):
        w_corr = self.w_correlation(ts_components)
        abs_w_corr = np.abs(w_corr)
        return abs_w_corr


    @lru_cache(maxsize=None)
    def _prepare_forecast(self,
                          use_components):

        use_components = np.array(use_components)

        U = np.concatenate([
            self.left_singular_vectors_[self._get_lidx(i):(self._get_ridx(i)-1), :]
            for i in range(self.P_)
        ], axis=0)
        U = U[:, use_components]

        W = np.array([
            self.left_singular_vectors_[self._get_ridx(i)-1, :]
            for i in range(self.P_)
        ])
        W = W[:, use_components]

        R = construct_forecasting_matrix(
            self.P_,
            W,
            U
        )
        return R



    def forecast(self,
                 timepoints_out,
                 timeseries_indices=None,
                 use_components=None):

        if use_components is None:
            use_components = np.arange(self.rank_)
        if isinstance(use_components, int):
            use_components = np.arange(use_components)

        recons = [diagonal_average_at_components(self.elementary_matrices_[ts_idx, :, :, :],
                                                 use_components)
                  for ts_idx in range(self.P_)]
        recons = np.array(recons)

        R = self._prepare_forecast(tuple(use_components))

        for t in range(timepoints_out):
            Z = recons[:, (-self.L_ + 1):]
            Z = Z.ravel()[:, np.newaxis]
            fc = np.dot(R, Z)
            recons = np.concatenate([recons, fc], axis=1)

        forecasted = recons[:, -timepoints_out:]
        if timeseries_indices is not None:
            timeseries_indices = np.atleast_1d(timeseries_indices)
            forecasted = forecasted[timeseries_indices, :]

        return forecasted
