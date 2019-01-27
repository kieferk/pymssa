import numpy as np
from numpy.linalg import matrix_rank

from pprint import pprint

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import hankel

from functools import partial, lru_cache, reduce
from tqdm.autonotebook import tqdm

from .optimized import *
from .ops import *

from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import explained_variance_score




class MSSA:
    '''Multivariate Singular Spectrum Analysis

    Implements MSSA decomposition and (recurrent) forecasting using MSSA. This
    implementation uses the vertical (V-MSSA) rather than horizontal (H-MSSA)
    structure for the trajectory matrix.


    Parameters
    ----------
    window_size : int | None
        The window size parameter controls the dimensionality of the trajectory
        matrices constructed for each timeseries (and then stacked). Timeseries
        are converted into trajectory matrices through "hankelization", where
        columns and rows represent different "windows" of the timeseries,
        incrementing across the timeseries. With window_size = L, the resulting
        trajectory matrix of a timeseries vector (N) will be of shape (L, K), where
        K = N - L + 1. As such, window_size should be no greater than N // 2. If
        left as None, MSSA will select the maximum possible window size.

        Note that with a multivariate timeseries input matrix (N, P), the resulting
        trajectory matrix stacked vertically will be of shape (P * L, K).

        The window size parameter can have a significant impact on the quality of
        the MSSA decomposition and forecasting. Some recommend that window
        size should be as large as possible to capture the most signal
        in the data, but there does not seem to be general agreement on a "best"
        window size. The author of the MSSA algorithm states in one of her papers
        that it is best to try many different window size parameters to see what
        works best with your data. If you have an idea of what frequency signal
        will occur in your data, try out window sizes that are multiples of that
        frequency (e.g. 24, 36, 48 if you have monthly data).

    n_components: int | None | 'variance_threshold' | 'parallel_analysis' | 'svht'
        Argument specifing the number of components to keep from the SVD decomposition.
        This is the equivalent of the n_components parameter in sklearn's PCA,
        for example. If None, the maximum number of (non-zero singular value)
        components will be selected.

        There are a few autmatic options for component selection:
        - 'svht'
            Select components using the Singular Value Hard Thresholding
            formula. This is the default setting. For more details on this
            formula please see this paper: https://arxiv.org/pdf/1305.5870.pdf
        - 'parallel_analysis'
            Performs parallel analysis to select the number of components that
            outperform a user-specified percentile threshold of noise components
            from randomly generated datasets of the same shape. Parallel analysis
            is a gold standard method for selecting a number of components in
            principal component analysis, which MSSA is closely related to.
            Eigenvalue noise threshold is set via the `pa_percentile_threshold`
            argument. Note that this procedure can be very slow depending on
            the size of your data.
        - 'variance_threshold'
            Select the number of components based on a variance explained percent
            threshold. The threshold cutoff is specified by the argument
            `variance_explained_threshold`

    variance_explained_threshold : float | None
        If `n_components = 'variance_threshold'`, this argument controls the
        cutoff for keeping components based on cumulative variance explained. This
        must be a float between 0 and 1. A value of 0.95, for example, will
        keep the number of components that explain 95 percent of the variance.
        This has no effect unless 'variance_threshold' is the selected method for
        `n_components`.

    pa_percentile_threshold : float | None
        If `n_components = 'parallel_analysis'`, this specifies the percentile
        of noise eigenvalues that must be exceeded by the real eigenvalues for
        components to be kept. Should be a number between 0 and 100. This has no
        effect unless 'parallel_analysis' is selected for `n_components`.

    svd_method : str
        Can be one of:
        - 'randomized'
            The default. Uses the `randomized_svd` method from scikit-learn to
            perform the singular value decomposition step. It is highly recommended
            that you keep this argument as 'randomized', especially if you are
            dealing with large data.
        - 'exact'
            Performs exact SVD via numpy.linalg.svd. This should be OK for small
            or even medium size datasets, but is not recommended.

    varimax : bool
        [EXPERIMENTAL] If `True`, performs a structured varimax rotation on the
        left singular vectors following the SVD decomposition step in the
        MSSA algorithm. This should be used with caution as the code is experimental.
        The idea of applying structured varimax is to better separate the components
        for the multiple timeseries fit by MSSA. See this presentation for
        more information on the structured varimax rotation applied to MSSA:
        http://200.145.112.249/webcast/files/SeminarMAR2017-ICTP-SAIFR.pdf

    verbose : bool
        Verbosity flag. If true, will print out status updates during the fit
        procedure.


    Attributes
    ----------
    These attributes will become available after fitting.

    N_ : int
        Observations in timeseries.
    P_ : int
        Number of timeseries.
    L_ : int
        Window size of trajectory matrices.
    K_ : int
        Column dimension of trajectory matrices.
    rank_ : int
        The selected rank (number of components kept)
    left_singular_vectors_ : numpy.ndarray
        The left singular vectors from the decomposition of the covariance of
        trajectory matrices via SVD.
    singular_values_ : numpy.ndarray
        Singular values from SVD
    explained_variance_ : numpy.ndarray
        The explained variance of the SVD components
    explained_variance_ratio_ : numpy.ndarray
        Percent of explained variance for each component
    components_ : numpy.ndarray
        The MSSA components. This is the result of the decomposition and
        reconstruction via diagonal averaging. The sum of all the components
        for a timeseries (without reducing number of components) will perfectly
        reconstruct the original timeseries.
        The dimension of this matrix is (P, N, rank), where P is the number
        of timeseries, N is the number of observations, and rank is the
        number of components selected to keep.
    component_ranks_ : numpy.ndarray
        This matrix shows the rank of each component per timeseries according
        to the reconstruction error. This is a (rank, P) matrix, with rank
        being the number of components and P the number of timeseries. For
        example, if component_ranks_[0, 0] = 3, this would mean that the
        3rd component accounts for the most variance for the first timeseries.
    component_ranks_explained_variance_ : numpy.ndarray
        This shows the explained variance percent for the ranked components
        per timeseries. Like component_ranks_, this is a (rank, P) matrix.
        The values in this matrix correspond to the percent of variance
        explained by components per timeseries in rank order of their
        efficiency in reconstructing the timeseries.
    '''


    def __init__(self,
                 window_size=None,
                 n_components='svht',
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


    def _apply_structured_varimax(self,
                                  left_singular_vectors,
                                  singular_values,
                                  P,
                                  L,
                                  gamma=1,
                                  tol=1e-6,
                                  max_iter=1000):
        '''
        [EXPERIMENTAL]
        Applies the structured varimax rotation to the left singular vectors
        and singular values. For more information on this procedure in MSSA please
        see this slideshow:
        http://200.145.112.249/webcast/files/SeminarMAR2017-ICTP-SAIFR.pdf
        '''

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



    def _parallel_analysis_component_selection(self,
                                               timeseries,
                                               L,
                                               K,
                                               rank,
                                               singular_values,
                                               iterations=100):
        '''
        Performs parallel analysis to help select the appropriate number of MSSA
        components to keep. The algorithm follows these steps:
        1. Calculate the eigenvalues via SVD/PCA on your real dataset.
        2. For a given number of iterations:
            3. Construct a random noise matrix the same shape as your real data.
            4. Perform decomposition of the random noise data.
            5. Calculate the eigenvalues for the noise data and track them per
               iteration.
        6. Calculate the percentile at a user-specified threshold of the noise
           eigenvalues at each position.
        7. Select only the number of components in the real data whose eigenvalues
           exceed those at the specified percentile of the noise eigenvalues.
        '''

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



    def _calculate_optimal_reconstruction_orders(self,
                                                 timeseries,
                                                 components):
        '''Calculates the optimal component ordering for reconstructing
        each of the timeseries. This is done by simply ranking the components
        in terms of how much variance they explain for each timeseries in the
        original data.
        '''

        optimal_orders = optimal_component_ordering(
            timeseries,
            components
        )

        optimal_orders = optimal_orders.astype(int)

        order_explained_variance = np.zeros_like(optimal_orders).astype(float)
        for ts_idx in range(timeseries.shape[1]):
            ts_comp = components[ts_idx, :, :]
            ts_comp = ts_comp[:, optimal_orders[:, ts_idx]]
            # ts_comp = np.cumsum(ts_comp, axis=1)

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
        '''Performs MSSA decomposition on a univariate or multivariate timeseries.
        Multivariate timeseries should have observations in rows and timeseries
        indices in columns.

        After fitting, many attributes become available to the user:
        N_ : int
            Observations in timeseries.
        P_ : int
            Number of timeseries.
        L_ : int
            Window size of trajectory matrices.
        K_ : int
            Column dimension of trajectory matrices.
        rank_ : int
            The selected rank (number of components kept)
        left_singular_vectors_ : numpy.ndarray
            The left singular vectors from the decomposition of the covariance of
            trajectory matrices via SVD.
        singular_values_ : numpy.ndarray
            Singular values from SVD
        explained_variance_ : numpy.ndarray
            The explained variance of the SVD components
        explained_variance_ratio_ : numpy.ndarray
            Percent of explained variance for each component
        components_ : numpy.ndarray
            The MSSA components. This is the result of the decomposition and
            reconstruction via diagonal averaging. The sum of all the components
            for a timeseries (without reducing number of components) will perfectly
            reconstruct the original timeseries.
            The dimension of this matrix is (P, N, rank), where P is the number
            of timeseries, N is the number of observations, and rank is the
            number of components selected to keep.
        component_ranks_ : numpy.ndarray
            This matrix shows the rank of each component per timeseries according
            to the reconstruction error. This is a (rank, P) matrix, with rank
            being the number of components and P the number of timeseries. For
            example, if component_ranks_[0, 0] = 3, this would mean that the
            3rd component accounts for the most variance for the first timeseries.
        component_ranks_explained_variance_ : numpy.ndarray
            This shows the explained variance percent for the ranked components
            per timeseries. Like component_ranks_, this is a (rank, P) matrix.
            The values in this matrix correspond to the percent of variance
            explained by components per timeseries in rank order of their
            efficiency in reconstructing the timeseries.

        Parameters
        ----------
        timeseries : numpy.ndarray | pandas.DataFrame | pandas.Series
            The timeseries data to be decomposed. This will be converted to
            a numpy array if it is in pandas format.
        '''

        timeseries = getattr(timeseries, 'values', timeseries)

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

            self.left_singular_vectors_, self.singular_values_ = self._apply_structured_varimax(
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

            self.rank_ = self._parallel_analysis_component_selection(
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

        ranks, rank_exp_var = self._calculate_optimal_reconstruction_orders(
            self.timeseries_,
            self.components_
        )
        self.component_ranks_ = ranks
        self.component_ranks_explained_variance_ = rank_exp_var

        self.component_groups_ = {
            ts_idx:[i for i in range(self.components_.shape[2])]
            for ts_idx in range(self.P_)
        }

        return self


    @property
    def hankel_weights_(self):
        '''The hankel weights are used to calculate the weighted correlation
        between components'''
        weights = construct_hankel_weights(
            self.L_,
            self.K_,
            self.N_
        )
        weights = weights.astype(float)
        return weights


    def w_correlation(self, ts_components):
        '''Calculates the w-correlation (weighted correlation) between timeseries
        components according to the hankelization weights. The weighting is
        required for an appropriate correlation measure since in the trajectory
        matrix format of a timeseries observations end up repeated multiple times.
        Observations that are in fewer "windows" of the trajectory matrix
        are downweighted relative to those that appear in many windows.'''
        weights = self.hankel_weights_
        w_corr = hankel_weighted_correlation(
            ts_components,
            weights
        )
        return w_corr

    @property
    def grouped_components_(self):
        if getattr(self, 'component_groups_', None) is None:
            return None

        _cgrouped = {
            ts_idx:np.concatenate([
                self.components_[ts_idx, :, np.atleast_1d(group)].T.sum(axis=1)[:, np.newaxis]
                for group in ts_cgroups
            ], axis=1)
            for ts_idx, ts_cgroups in self.component_groups_.items()
        }

        return _cgrouped


    def _validate_component_group_assignment(self,
                                             timeseries_index,
                                             groups):
        if getattr(self, 'component_groups_', None) is None:
            raise Exception('MSSA must be fit before assigning component groups.')

        if timeseries_index not in self.component_groups_:
            raise Exception('timeseries_index not in {}'.format(self.component_groups_.keys()))

        if not isinstance(groups, (list, tuple, np.ndarray)):
            raise Exception('groups must be a list of lists (or int), with each sub-list component indices')

        for group in groups:
            group = np.atleast_1d(group)
            for ind in group:
                if ind not in np.arange(self.components_.shape[2]):
                    raise Exception('Component index {} not in valid range'.format(ind))

        return True


    def set_component_groups(self,
                             component_groups_dict):
        '''Method to assign component groupings via a dictionary. The dictionary
        must be in the format:

        `{timeseries_index:groups}`

        Where `timeseries_index` is the column index of the timeseries, and
        groups is a list of lists where each sublist contains indices for the
        components in that particular group.

        For example, if you were updating the component groupings for the first
        two timeseries it might look something like this:

        `{
            0:[
                [0,1,2],
                [3],
                [4,5],
                [6,7,8]
            ],
            1:[
                [0],
                [1,2],
                [3],
                [4,5,6]
            ]
        }`

        The passed in dictionary will update the `component_groups_` attribute.
        Note that this function will raise an exception if the fit method has
        not been run yet, since there are no components until decomposition occurs.

        The `component_groups_` attribute defaults to one component per group
        after fitting (as if all components are independent).

        The `grouped_components_` attribute is a dictionary with timeseries
        indices as keys and the grouped component matrix as values. These matrices
        are the actual data representation of the groups that you specify in
        `component_groups_`. If you change `component_groups_`, the `grouped_components_`
        attribute will automatically update to reflect this.

        Parameters
        ----------
        component_group_dict : dict
            Dictionary with timeseries index as keys and list-of-list component
            index groupings as values. Updates the `component_groups_` and
            `grouped_components_` attributes.

        '''
        if not isinstance(component_groups_dict, dict):
            raise Exception('Must provide a dict with ts_index:groups as key:value pairs')

        for ts_idx, groups in component_groups_dict.items():
            _ = self._validate_component_group_assignment(ts_idx, groups)

        self.component_groups_.update(component_groups_dict)

        return self


    def set_ts_component_groups(self,
                                timeseries_index,
                                groups):
        '''Method to assign component groupings via a timeseries index and a
        list of lists, where each sublist is indices of the components for that
        group. This is an alternative to the `set_component_groups` function.

        For example if you were updating component 1 it may look something like
        this:

        `timeseries_index = 1
        groups = [
            [0],
            [1,2],
            [3],
            [4,5,6]
        ]

        mssa.set_ts_component_groups(timeseries_index, groups)
        }`

        This will update the `component_groups_` attribute with the new groups
        for the specified timeseries index.

        Note that this function will raise an exception if the fit method has
        not been run yet, since there are no components until decomposition occurs.

        The `component_groups_` attribute defaults to one component per group
        after fitting (as if all components are independent).

        The `grouped_components_` attribute is a dictionary with timeseries
        indices as keys and the grouped component matrix as values. These matrices
        are the actual data representation of the groups that you specify in
        `component_groups_`. If you change `component_groups_`, the `grouped_components_`
        attribute will automatically update to reflect this.

        Parameters
        ----------
        timeseries_index : int
            Column index of the timeseries to update component groupings for.
        groups : list
            List of lists, where each sub-list is indices for components in
            that particular group.
        '''

        _ = self._validate_component_group_assignment(timeseries_index, groups)
        self.component_groups_[timeseries_index] = groups

        return self



    def forecast(self,
                 timepoints_out,
                 timeseries_indices=None,
                 use_components=None):
        '''Forecasts out a number of timepoints using the recurrent forecasting
        formula.

        Parameters
        ----------
        timepoints_out : int
            How many timepoints to forecast out from the final observation given
            to fit in MSSA.
        timeseries_indices : None | int | numpy.ndarray
            If none, forecasting is done for all timeseries. If an int or array
            of integers is specified, the forecasts for the timeseries at those
            indices is performed. (In reality this will always forecast for all
            timeseries then simply use this to filter the results at the end.)
        use_components : None | int | numpy.ndarray
            Components to use in the forecast. If None, all components will be
            used. If an int, that number of top components will be selected (e.g
            if `use_components = 10`, the first 10 components will be used). If
            a numpy array, those compoents at the specified indices will be used.
        '''

        if use_components is None:
            use_components = np.arange(self.components_.shape[2])
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
