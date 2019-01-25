import numpy as np
from numpy.linalg import matrix_rank

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import hankel

from functools import partial, lru_cache, reduce
from tqdm.autonotebook import tqdm

from .optimized import *
from .ops import *

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
            rnorm_trajectory_matrix = ts_matrix_to_trajectory_matrix(
                ts_rnorm,
                L,
                K
            )

            # decompose the noise trajectory matrix
            U, s, V, rank = decompose_trajectory_matrix(
                rnorm_trajectory_matrix,
                rank,
                svd_method=self.svd_method
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



    def calculate_optimal_reconstruction_orders(self,
                                                timeseries,
                                                components):

        optimal_orders = optimal_component_ordering(
            timeseries,
            components
        )

        optimal_orders = optimal_orders.astype(int)

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

        self.trajectory_matrix_ = ts_matrix_to_trajectory_matrix(
            self.timeseries_,
            self.L_,
            self.K_
        )

        if self.verbose:
            print('Trajectory matrix shape:', self.trajectory_matrix_.shape)

        if self.verbose:
            print('Decomposing trajectory covariance matrix with SVD')

        U, s, V, rank = decompose_trajectory_matrix(
            self.trajectory_matrix_,
            self.K_,
            svd_method=self.svd_method
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

        exp_var, exp_var_ratio = sv_to_explained_variance_ratio(
            self.singular_values_,
            self.N_
        )
        self.explained_variance_ = exp_var
        self.explained_variance_ratio_ = exp_var_ratio

        if self.n_components == 'svht':
            self.rank_ = singular_value_hard_threshold(
                self.singular_values_,
                rank=self.rank_
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
            print('Constructing components')

        self.components_ = incremental_component_reconstruction(
            self.trajectory_matrix_,
            self.left_singular_vectors_,
            self.singular_values_,
            self.rank_,
            self.P_,
            self.N_,
            self.L_
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
        weights = construct_hankel_weights(
            self.L_,
            self.K_,
            self.N_
        )
        weights = weights.astype(float)
        return weights


    def w_correlation(self, ts_components):
        weights = self.hankel_weights_
        w_corr = hankel_weighted_correlation(
            ts_components,
            weights
        )
        return w_corr


    def forecast(self,
                 timepoints_out,
                 timeseries_indices=None,
                 use_components=None):

        if use_components is None:
            use_components = np.arange(left_singular_vectors.shape[1])
        elif isinstance(use_components, int):
            use_components = np.arange(use_components)

        forecasted = vmssa_recurrent_forecast(
            timepoints_out,
            self.components_,
            self.left_singular_vectors_,
            self.P_,
            self.L_,
            use_components=use_components
        )

        if timeseries_indices is not None:
            timeseries_indices = np.atleast_1d(timeseries_indices)
            forecasted = forecasted[timeseries_indices, :]

        return forecasted
