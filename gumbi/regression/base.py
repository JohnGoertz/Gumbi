import warnings
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from itertools import product

from gumbi.utils.misc import assert_in, assert_is_subset
from gumbi.utils.gp_utils import get_ℓ_prior
from gumbi.aggregation import DataSet
from gumbi.arrays import *
from gumbi.arrays import ParameterArray as parray
from gumbi.arrays import UncertainParameterArray as uparray
from gumbi.arrays import MVUncertainParameterArray as mvuparray

__all__ = ['Regressor']


class Regressor(ABC):
    r"""Surface learning and prediction.

    A Regressor is built from a dataframe in the form of a :class:`DataSet` object. This is stored as
    :attr:`tidy`. The model inputs are constructed by filtering this dataframe, extracting column values, and
    converting these to numerical input coordinates. Each subclass defines at least `build_model`, `fit`, and `predict_points`
    methods in addition to subclass-specific methods.

    Dimensions fall into several categories:

    * Filter dimensions, those with only one level, are used to subset the dataframe but are not included as explicit
      inputs to the model. These are not specified explicitly, but rather any continuous or categorical dimension with only one
      level is treated as a filter dimension.
    * Continuous dimensions are treated as explicit coordinates and given a Radial Basis Function kernel

      * Linear dimensions (which must be a subset of `continuous_dims`) have an additional linear kernel.

    * Coregion dimensions imply a distinct but correlated output for each level

      * If more than one output is specified, ``self.out_col`` is treated as a categorical dim.

    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    outputs : str or list of str, default None
        Name(s) of output(s) to learn. If ``None``, uses all values from ``outputs`` attribute of *dataset*.
    seed : int
        Random seed

    Attributes
    ----------
    data : DataSet
        Data for fitting.
    outputs : list of str, optional
        Name(s) of output(s) to learn.
    seed : int
        Random seed
    continuous_dims : list of str
        Columns of dataframe used as continuous dimensions
    linear_dims : list of str
        Subset of continuous dimensions to apply an additional linear kernel.
    continuous_levels : dict
        Values considered within each continuous column as ``{dim: [level1, level2]}``
    continuous_coords : dict
        Numerical coordinates of each continuous level within each continuous dimension as ``{dim: {level: coord}}``
    categorical_dims : list of str
        Columns of dataframe used as categorical dimensions
    categorical_levels : dict
        Values considered within each categorical column as ``{dim: [level1, level2]}``
    categorical_coords : dict
        Numerical coordinates of each categorical level within each categorical dimension as ``{dim: {level: coord}}``
    additive : bool
        Whether to treat categorical dimensions as additive or joint
    filter_dims : dict
        Dictionary of column-value pairs used to filter dataset before fitting
    X : array
        A 2D tall array of input coordinates.
    y : array
        A 1D vector of observations
    """

    def __init__(self, dataset: DataSet, outputs=None, seed=2021):
        if not isinstance(dataset, DataSet):
            raise TypeError('Learner instance must be initialized with a DataSet object')

        self.data = dataset
        self.stdzr = dataset.stdzr
        outputs = outputs if outputs is not None else dataset.outputs
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        self.out_col = dataset.names_column
        self.seed = seed

        self.continuous_dims = []
        self.linear_dims = []
        self.continuous_levels = {}
        self.continuous_coords = {}
        self.categorical_dims = []
        self.categorical_levels = {}
        self.categorical_coords = {}
        self.additive = False
        self.model_specs = {}

        self.X = None
        self.y = None

        self.grid_vectors = None
        self.grid_parray = None
        self.grid_points = None
        self.ticks = None

        self.predictions = None

    ################################################################################
    # Model building and fitting
    ################################################################################

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Defined by subclass

        See Also
        --------
        :meth:`GP.fit`
        :meth:`GLM.fit`
        """
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """Defined by subclass

        See Also
        --------
        :meth:`GP.build_model`
        :meth:`GLM.build_model`
        """
        pass

    ################################################################################
    # Properties and convenience methods
    ################################################################################

    def parray(self, **kwargs) -> parray:
        """Creates a parray with the current instance's stdzr attached"""
        return parray(stdzr=self.stdzr, **kwargs)

    def uparray(self, name: str, μ: np.ndarray, σ2: np.ndarray, **kwargs) -> uparray:
        """Creates a uparray with the current instance's stdzr attached"""
        return uparray(name, μ, σ2, stdzr=self.stdzr, **kwargs)

    def mvuparray(self, *uparrays, cor, **kwargs) -> mvuparray:
        """Creates a uparray with the current instance's stdzr attached"""
        return mvuparray(*uparrays, cor=cor, stdzr=self.stdzr, **kwargs)

    @property
    def dims(self) -> list:
        """List of all dimensions under consideration"""
        return self.continuous_dims + self.categorical_dims

    @property
    def levels(self) -> dict:
        """Dictionary of values considered within each dimension as ``{dim: [level1, level2]}``"""
        return self.continuous_levels | self.categorical_levels

    @property
    def coords(self) -> dict:
        """ Dictionary of numerical coordinates of each level within each dimension as ``{dim: {level: coord}}``"""
        return self.continuous_coords | self.categorical_coords

    ################################################################################
    # Preprocessing
    ################################################################################

    def specify_model(self, outputs=None, linear_dims=None, continuous_dims=None, continuous_levels=None, continuous_coords=None,
                      categorical_dims=None, categorical_levels=None, additive=False):
        """Checks for consistency among dimensions and levels and formats appropriately.

        Parameters
        ----------
        outputs : str or list of str, default None
            Name(s) of output(s) to learn. If ``None``, :attr:`outputs` is used.
        linear_dims : str or list of str, optional
            Subset of continuous dimensions to apply an additional linear kernel. If ``None``, defaults to ``['Y','X']``.
        continuous_dims : str or list of str, optional
            Columns of dataframe used as continuous dimensions
        continuous_levels : str, list, or dict, optional
            Values considered within each continuous column as ``{dim: [level1, level2]}``
        continuous_coords : list or dict, optional
            Numerical coordinates of each continuous level within each continuous dimension as ``{dim: {level: coord}}``
        categorical_dims : str or list of str, optional
            Columns of dataframe used as categorical dimensions
        categorical_levels : str, list, or dict, optional
            Values considered within each categorical column as ``{dim: [level1, level2]}``
        additive : bool, default False
            Whether to treat categorical_dims as additive or joint (default)

        Returns
        -------
        self : :class:`GP`
        """

        # Ensure output is valid and format as list
        outputs = outputs if outputs is not None else self.outputs
        assert_is_subset(self.out_col, outputs, self.data.outputs)
        self.outputs = outputs if isinstance(outputs, list) else [outputs]

        # Ensure dimensions are valid and format as list
        self.continuous_dims = self._parse_dimensions(continuous_dims)
        self.linear_dims = self._parse_dimensions(linear_dims)
        self.categorical_dims = self._parse_dimensions(categorical_dims)
        if set(self.categorical_dims) & set(self.continuous_dims) != set():
            raise ValueError('Overlapping items in categorical_dims and continuous_dims')

        # Ensure levels are valid and format as dict
        self.continuous_levels = self._parse_levels(self.continuous_dims, continuous_levels)
        self.categorical_levels = self._parse_levels(self.categorical_dims, categorical_levels)

        # Add self.out_col to the end of the categorical list
        self.categorical_dims += [self.out_col]
        self.categorical_levels[self.out_col] = self.outputs

        # Move dims with only one level to separate list
        self.filter_dims = {}
        for dim in self.dims:
            levels = self.levels[dim]
            if len(levels) == 1:
                self.filter_dims[dim] = levels
                self.continuous_dims = [d for d in self.continuous_dims if d != dim]
                self.categorical_dims = [d for d in self.categorical_dims if d != dim]
                self.continuous_levels = {d: l for d, l in self.continuous_levels.items() if d != dim}
                self.categorical_levels = {d: l for d, l in self.categorical_levels.items() if d != dim}

        # Ensure coordinates are valid and format as dict-of-dicts
        self.continuous_coords = self._parse_coordinates(self.continuous_dims, self.continuous_levels, continuous_coords)
        self.categorical_coords = self._parse_coordinates(self.categorical_dims, self.categorical_levels, None)

        # Add 'X' and 'Y' to the beginning of the continuous list
        # if 'Y' not in self.continuous_dims:
        #     self.continuous_dims = ['Y'] + self.continuous_dims
        # if 'X' not in self.continuous_dims:
        #     self.continuous_dims = ['X'] + self.continuous_dims

        # self.continuous_levels | {dim: self.tidy.tidy[dim].unique() for dim in ['X', 'Y']} | self.continuous_levels}
        # self.continuous_coords | {dim: {level: level for level in self.continuous_levels[dim]} for dim in ['X', 'Y']} | self.continuous_coords}
        assert_is_subset('continuous dimensions', self.linear_dims, self.continuous_dims)
        self.additive = additive
        return self

    def _parse_dimensions(self,
                          dims: None or str or list) -> list:
        """Ensure dimensions are possible and formatted as list"""
        if dims is not None:
            assert self.out_col not in dims
            dims = dims if isinstance(dims, list) else [dims]
            assert_is_subset('columns', dims, self.data.tidy.columns)
        else:
            dims = []
        return dims

    def _parse_levels(self, dims: list, levels: None or str or list or dict) -> dict:
        """Perform consistency checks between dimensions and levels and format `levels` as dict"""
        if len(dims) != 0:
            if levels is None:
                # Use all levels of all dims
                levels = {dim: list(self.data.tidy[dim].unique()) for dim in dims}
            elif any(isinstance(levels, typ) for typ in [str, list]):
                # If only a single dim is supplied, convert levels to dictionary
                assert len(dims) == 1, 'Non-dict argument for `levels` only allowed if `len(dims)==1`'
                levels = levels if isinstance(levels, list) else [levels]
                levels = {dims[0]: levels}
            elif isinstance(levels, dict):
                # Ensure levels are specified as lists
                for d, v in levels.items():
                    if not isinstance(v, list):
                        levels[d] = [v]
                # Ensure each dimension specified by levels is valid
                if (bad := [dim for dim in levels.keys() if dim not in dims]):
                    raise KeyError(f'Dimensions {bad} specified in *levels not found in *dims')
                # Ensure each level is valid
                if (bad := {k: v for k, vs in levels.items() for v in vs if v not in self.data.tidy[k].unique()}):
                    raise ValueError(f'Values specified in *levels not found in tidy: {bad}')
                # Use all levels of remaining dims
                levels |= {dim: list(self.data.tidy[dim].unique()) for dim in dims if dim not in levels.keys()}
            else:
                raise TypeError('`levels` must be of type str, list, or dict')

            for dim in dims:
                assert_is_subset(f'data[{dim}]', levels[dim], self.data.tidy[dim])
        else:
            levels = {}
        return levels

    def _parse_coordinates(self, dims: list, levels: dict, coords: None or list or dict) -> dict:
        """Check for consistency between supplied dims/levels/coords or generate coords automatically"""
        if coords is not None:
            if isinstance(coords, dict):
                # Ensure all dim-level pairs in ``levels`` and ``coords`` match exactly
                level_tuples = [(dim, level) for dim, levels_list in levels.items() for level in levels_list]
                coord_tuples = [(dim, level) for dim, coord_dict in coords.items() for level in coord_dict.keys()]
                assert_is_subset('coordinates', coord_tuples, level_tuples)
                assert_is_subset('coordinates', level_tuples, coord_tuples)
            elif isinstance(coords, list):
                assert len(levels.keys()) == 1, \
                    'Non-dict argument for `continuous_coords` only allowed if `len(continuous_dims)==1`'
                dim = dims[0]
                assert len(coords) == len(levels[dim])
                coords = {dim: {level: coord for level, coord in zip(levels[dim], coords)}}
            else:
                raise TypeError('Coordinates must be of type list or dict')
            if not all(isinstance(coord, (int, float))
                       for coord_dict in coords.values()
                       for coord in coord_dict.values()):
                raise TypeError('Coordinates must be numeric')
        elif dims is not None and levels is not None:
            coords = {dim: self._make_coordinates(dim, levels_list) for dim, levels_list in levels.items()}
        else:
            coords = {}
        return coords

    def _make_coordinates(self, dim: str, levels_list: list) -> dict:
        """Generate numerical coordinates for each level in each dim under consideration"""

        df = self.data.tidy
        col = df[df[dim].isin(levels_list)][dim]

        if col.dtype in [np.float32, np.float64, np.int32, np.int64]:
            coords = {level: level for level in levels_list}
        else:
            coords = {level: col.astype('category').cat.categories.to_list().index(level) for level in levels_list}

        return coords

    def get_filtered_data(self, standardized=False, metric='mean'):
        """The portion of the dataset under consideration

        A filter is built by comparing the values in the unstandardized dataframe with those in :attr:`filter_dims`,
        :attr:`categorical_levels`, and :attr:`continuous_levels`, then the filter is applied to the standardized or
        unstandardized dataframe as indicated by the `standardized` input argument.

        Parameters
        ----------
        standardized : bool, default True
            Whether to return a subset of the raw tidy or the centered and scaled tidy
        metric : str, default 'mean'
            Which summary statistic to return (must be a value in the `Metric` column)

        Returns
        -------
        tidy : pd.DataFrame
        """

        df = self.data.tidy

        allowed = df.isin(self.filter_dims)[self.filter_dims.keys()].all(axis=1)
        if 'Metric' in df.columns:
            assert_in('Metric', metric, self.data.tidy['Metric'].unique())
            allowed &= df['Metric'] == metric
        for dim, levels in self.levels.items():
            allowed &= df[dim].isin(levels)

        return df[allowed] if not standardized else self.data.tidy.z[allowed]

    def get_structured_data(self, metric='mean'):
        """Formats input data and observations as parrays

        Parameters
        ----------
        metric : str, default 'mean'
            Which summary statistic to return (must be a value in the `Metric` column)

        Returns
        -------
        X : parray
            A multilayered column vector of input coordinates.
        y : parray
            A multilayered (1D) vector of observations

        See Also
        --------
        :meth:`get_filtered_data`

        """

        df = self.get_filtered_data(standardized=False, metric=metric)

        # Ensure same number of observations for every output (only possible if something broke)
        assert len(set(sum(df[self.out_col] == output) for output in self.outputs)) == 1

        # Assuming all parameters observed at the same points
        # Extract the model dimensions from the dataframe for one of the parameters
        dims = set(self.dims) - set([self.out_col])
        dim_values = {dim: df[df[self.out_col] == self.outputs[0]].replace(self.coords)[dim].values for dim in dims}
        X = self.parray(**dim_values, stdzd=False)

        # List of parrays for each output
        outputs = {output: df[df[self.out_col] == output]['Value'].values for output in self.outputs}
        y = self.parray(**outputs, stdzd=False)

        return X, y

    def get_shaped_data(self, metric='mean'):
        """Formats input data and observations as plain numpy arrays

        Parameters
        ----------
        metric : str, default 'mean'
            Which summary statistic to return (must be a value in the `Metric` column)

        Returns
        -------
        X : np.ndarray
            A tall matrix of input coordinates with shape (n_obs, n_dims).
        y : np.ndarray
            A (1D) vector of observations

        See Also
        --------
        :meth:`get_filtered_data`

        """

        self.X, self.y = self.get_structured_data(metric=metric)

        # Convert ParameterArray into plain numpy tall array
        if self.out_col in self.dims:
            ordered_outputs = {k: v for k, v in sorted(self.coords[self.out_col].items(), key=lambda item: item[1])}
            y = np.hstack([self.y.z[output+'_z'].values() for output in ordered_outputs.keys()])
            X = self.X[:, None]  # convert to column vector
            X = parray.vstack([X.add_layers(**{self.out_col: coord}) for coord in ordered_outputs.values()])
            X = np.atleast_2d(np.column_stack([X[dim].z.values().squeeze() for dim in self.dims]))
        else:
            y = self.y.z.values().squeeze()
            X = np.atleast_2d(np.column_stack([self.X[dim].z.values().squeeze() for dim in self.dims]))

        return X, y

    ################################################################################
    # Prediction
    ################################################################################

    @abstractmethod
    def predict(self, points_array, with_noise=True, **kwargs):
        """Defined by subclass.

        It is not recommended to call :meth:`predict` directly, since it requires a very specific formatting for inputs,
        specifically a tall array of standardized coordinates in the same order as :attr:`dims`. Rather, one of the
        convenience functions :meth:`predict_points` or :meth:`predict_grid` should be used, as these have a more
        intuitive input structure and format the tidy appropriately prior to calling :meth:`predict`.

        See Also
        --------
        :meth:`GP.predict`
        :meth:`GLM.predict`

        Returns
        -------
        prediction_mean, prediction_var : list of np.ndarray
            Mean and variance of predictions at each supplied points
        """
        pass

    def _check_has_prediction(self):
        """Does what it says on the tin"""
        if self.predictions is None:
            raise ValueError('No predictions found. Run self.predict_grid or related method first.')

    def _parse_prediction_output(self, output):
        if self.out_col in self.categorical_dims:
            # Multiple parameters are possible, determine which ones to predict
            if output is None:
                # predict all parameters in model
                output = self.categorical_levels[self.out_col]
            elif isinstance(output, list):
                assert_is_subset('Outputs', output, self.categorical_levels[self.out_col])
            elif isinstance(output, str):
                output = [output]
                assert_is_subset('Outputs', output, self.categorical_levels[self.out_col])
            else:
                raise ValueError('"output" must be list, string, or None')
        else:
            # If self.out_col is not in categorical_dims, it must be in filter_dims, and only one is possible
            output = self.filter_dims[self.out_col]

        return output

    def _prepare_points_for_prediction(self, points: ParameterArray, output):

        points = np.atleast_1d(points)
        assert points.ndim == 1
        assert set(self.dims) - set([self.out_col]) == set(points.names), \
            'All model dimensions must be present in "points" parray.'

        if self.out_col in self.categorical_dims:
            # Multiple parameters are possible, determine which ones to predict

            # Get model coordinates for each output to be predicted
            param_coords = [self.categorical_coords[self.out_col][p] for p in output]

            # Convert input points to tall array and tile once for each output, adding the respective coordinate
            tall_points = parray.vstack([points.add_layers(**{self.out_col: coord})[:, None] for coord in param_coords])
        else:
            # If self.out_col is not in categorical_dims, it must be in filter_dims, and only one is possible
            # Convert input points to tall array
            param_coords = None
            tall_points = points[:, None]

        # Combine standardized coordinates into an ordinary tall numpy array for prediction
        points_array = np.hstack([tall_points[dim].z.values() for dim in self.dims])

        return points_array, tall_points, param_coords

    def predict_points(self, points, output=None, with_noise=True, **kwargs):
        """Make predictions at supplied points

        Parameters
        ----------
        points : ParameterArray
            1-D ParameterArray vector of coordinates for prediction, must have one layer per ``self.dims``
        output : str or list of str, optional
            Variable for which to make predictions
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error
        **kwargs
            Additional keyword arguments passed to subclass-specific :meth:`predict` method

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a `uparray`
        """

        output = self._parse_prediction_output(output)
        points_array, tall_points, param_coords = self._prepare_points_for_prediction(points, output=output)

        # Prediction means and variance as a list of numpy vectors
        pred_mean, pred_variance = self.predict(points_array, with_noise=with_noise, **kwargs)
        self.predictions_X = points

        # Store predictions in appropriate structured array format
        if len(output) == 1:
            # Predicting one output, return an UncertainParameterArray
            self.predictions = self.uparray(output[0], pred_mean, pred_variance, stdzd=True)
        else:
            # Predicting multiple parameters, return an MVUncertainParameterArray
            # First split prediction into UncertainParameterArrays
            uparrays = []
            for i, name in enumerate(output):
                idx = (tall_points[self.out_col].values() == param_coords[i]).squeeze()
                μ = pred_mean[idx]
                σ2 = pred_variance[idx]
                uparrays.append(self.uparray(name, μ, σ2, stdzd=True))

            # Calculate the correlation matrix from the hyperparameters of the coregion kernel
            W = self.MAP[f'W_{self.out_col}'][param_coords, :]
            κ = self.MAP[f'κ_{self.out_col}'][param_coords]
            B = W @ W.T + np.diag(κ)  # covariance matrix
            D = np.atleast_2d(np.sqrt(np.diag(B)))  # standard deviations
            cor = B / (D.T @ D)  # correlation matrix

            # Store predictions as MVUncertainParameterArray
            self.predictions = self.mvuparray(*uparrays, cor=cor)

        return self.predictions

    def prepare_grid(self, limits=None, at=None, resolution=100):
        """Prepare unobserved input coordinates for specified continuous dimensions.

        Parameters
        ----------
        limits : ParameterArray
            List of min/max values as a single parray with one layer for each of a subset of `continuous_dims`.
        at : ParameterArray
            A single parray of length 1 with one layer for each remaining `continuous_dims` by name.
        ticks : dict
        resolution : dict or int
            Number of points along each dimension, either as a dictionary or one value applied to all dimensions

        Returns
        -------

        """

        # Remove any previous predictions to avoid confusion
        self.predictions = None
        self.predictions_X = None

        ##
        ## Check inputs for consistency and completeness
        ##

        # Ensure "at" is supplied correctly
        if at is None:
            at = self.parray(none=[])
        elif not isinstance(at, ParameterArray):
            raise TypeError('"at" must be a ParameterArray')
        elif at.ndim != 0:
            raise ValueError('"at" must be single point, potentially with multiple layers')

        # Ensure a grid can be built
        at_dims = set(at.names)
        continuous_dims = set(self.continuous_dims)
        limit_dims = continuous_dims-at_dims

        # If there are no remaining dimensions
        if limit_dims == set():
            raise ValueError('At least one dimension must be non-degenerate to generate grid.')

        # If no limits are supplied
        if limits is None:
            # Fill limits with default values
            limits = self.parray(**{dim: [-2.5, +2.5] for dim in self.dims if dim in limit_dims}, stdzd=True)
        else:
            # Append default limits to `limits` for unspecified dimensions
            if not isinstance(limits, ParameterArray):
                raise TypeError('"limits" must be a ParameterArray')
            remaining_dims = limit_dims-set(limits.names)
            if remaining_dims:
                dflt_limits = self.parray(**{dim: [-2.5, +2.5] for dim in remaining_dims}, stdzd=True)
                limits = limits.add_layers(**dflt_limits.as_dict())

        # Ensure all dimensions are specified without conflicts
        limit_dims = set(limits.names)

        if limit_dims.intersection(at_dims):
            raise ValueError('Dimensions specified via "limits" and in "at" must not overlap.')
        elif not continuous_dims.issubset(at_dims.union(limit_dims)-set(['none'])):
            raise ValueError('Not all continuous dimensions are specified by "limits" or "at".')

        # Format "resolution" as dict if necessary
        if isinstance(resolution, int):
            resolution = {dim: resolution for dim in self.continuous_dims}
        elif not isinstance(resolution, dict):
            raise TypeError('"resolution" must be a dictionary or an integer')
        else:
            assert_is_subset('continuous dimensions', resolution.keys(), self.continuous_dims)


        ##
        ## Build grids
        ##

        # Store a dictionary with one 1-layer 1-D parray for the grid points along each dimension
        # Note they may be different sizes
        grid_vectors = {dim:
            self.parray(
                **{dim: np.linspace(*limits[dim].z.values(), resolution[dim])[:, None]},
                stdzd=True)
            for dim in limit_dims}

        # Create a single n-layer n-dimensional parray for all evaluation points
        grids = np.meshgrid(*[grid_vectors[dim] for dim in self.dims if dim in limit_dims])
        grid_parray = self.parray(**{array.names[0]: array.values() for array in grids})

        # Add values specified in "at" to all locations in grid_parray
        if at.names != ['none']:
            at_arrays = {dim: np.full(grid_parray.shape, value) for dim, value in at.as_dict().items()}
            grid_parray = grid_parray.add_layers(**at_arrays)

        # Store dimensions along which grid was formed, ensuring the same order as self.dims
        self.prediction_dims = [dim for dim in self.dims if dim in limit_dims]
        self.grid_vectors = grid_vectors
        self.grid_parray = grid_parray
        self.grid_points = self.grid_parray.ravel()

        return grid_parray

    def predict_grid(self, output=None, categorical_levels=None, with_noise=True, **kwargs):
        """Make predictions and reshape into grid.

        If the model has :attr:`categorical_dims`, a specific level for each dimension must be specified as key-value
        pairs in `categorical_levels`.

        Parameters
        ----------
        output : str or list of str, optional
            Variable(s) for which to make predictions
        categorical_levels : dict, optional
            Level for each :attr:`categorical_dims` at which to make prediction
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a grid with len(:attr:`continuous_dims`) dimensions
        """

        if self.grid_points is None:
            raise ValueError('Grid must first be specified with `prepare_grid`')

        points = self.grid_points
        if self.categorical_dims:
            points = self.append_categorical_points(points, categorical_levels=categorical_levels)

        self.predict_points(points, output=output, with_noise=with_noise, **kwargs)
        self.predictions = self.predictions.reshape(self.grid_parray.shape)
        self.predictions_X = self.predictions_X.reshape(self.grid_parray.shape)

        return self.predictions

    def append_categorical_points(self, continuous_parray, categorical_levels):
        """Appends coordinates for the supplied categorical dim-level pairs to tall array of continuous coordinates.

        Parameters
        ----------
        continuous_points : ParameterArray
            Tall :class:`ParameterArray` of coordinates, one layer per continuous dimension
        categorical_levels : dict
            Single level for each :attr:`categorical_dims` at which to make prediction

        Returns
        -------
        points : ParameterArray
            Tall `ParameterArray` of coordinates, one layer per continuous and categorical dimension
        """

        if categorical_levels is not None:
            if set(categorical_levels.keys()) != (set(self.categorical_dims) - set([self.out_col])):
                raise AttributeError('Must specify level for every categorical dimension')

            points = continuous_parray.fill_with(**{dim: self.categorical_coords[dim][level]
                                                 for dim, level in categorical_levels.items()})
        else:
            points = continuous_parray
        return points

    ################################################################################
    # Proposals
    ################################################################################

    def propose(self, target, acquisition='EI'):
        """Bayesian Optimization with Expected Improvement acquisition function"""
        if self.predictions is None:
            raise ValueError('No predictions to make proposal from!')
        assert_in(acquisition, ['EI', 'PD'])
        output = self.predictions.name

        df = self.get_filtered_data(standardized=False)
        df = df[df[self.out_col] == output]
        observed = self.parray(**{output: df['Values']}, stdzd=False)

        target = self.parray(**{output: target}, stdzd=False)
        best_yet = np.min(np.sqrt(np.mean(np.square(observed.z - target.z))))

        if acquisition == 'EI':
            self.proposal_surface = self.predictions.z.EI(target.z, best_yet.z)
        elif acquisition == 'PD':
            self.proposal_surface = self.predictions.z.nlpd(target.z)

        self.proposal_idx = np.argmax(self.proposal_surface)
        self.proposal = self.predictions_X.ravel()[self.proposal_idx]

        return self.proposal

    ################################################################################
    # Evaluation
    ################################################################################

    def cross_validate(self, unit=None, *, n_train=None, pct_train=None, train_only=None, warm_start=True, seed=None,
                       errors='natural', **MAP_kws):
        """Fits model on random subset of tidy and evaluates accuracy of predictions on remaining observations.

        This method finds unique combinations of values in the columns specified by ``dims``, takes a random subset of
        these for training, and evaluates the predictions made for the remaining tidy.

        Notes
        -----
        :meth:`cross_validate` is *reproducibly random* by default. In order to evaluate different test/train subsets of
        the same size, you will need to set the `seed` explicitly.

        Specifying *unit* changes the interpretation of *n_train* and *pct_train*: rather than the number
        or fraction of all individual observations to be included in the training set, these now represent the number
        of distinct entities in the *unit* column from the wide-form dataset.

        Criteria in *train_only* are enforced before grouping observations by *unit*. If *train_only* and *unit* are
        both specified, but the *train_only* criteria encompass only some observations of a given entity in *unit*,
        this could lead to expected behavior.

        Similarly, if *warm_start* and *unit* are both specified, but a given entity appears in multiple categories
        from any of the :attr:`categorical_dims`, this could lead to expected behavior. It is recommended to set
        *warm_start* to `False` if this is the case.

        Parameters
        ----------
        unit : list of str
            Columns from which to take unique combinations as training and testing sets. This could be useful when the
            data contains multiple (noisy) observations for each of several distinct entities.
        n_train : int, optional
            Number of training points to use. Exactly one of `n_train` and `pct_train` must be specified.
        pct_train : float, optional
            Percent of training points to use. Exactly one of `n_train` and `pct_train` must be specified.
        train_only : dict, optional
            Specifications for observations to be always included in the training set. This will select all rows of the
            wide-form dataset which *exactly* match *all* criteria.
        warm_start : bool, default True
            Whether to include a minimum of one observation for each level in each `categorical_dim` in the training set.
        seed : int, optional
            Random seed
        errors : {'natural', 'standardized', 'transformed'}
            "Space" in which to return prediction errors
        **MAP_kws
            Additional

        Returns
        -------
        dict
            Dictionary with nested dictionaries 'train' and 'test', both containing fields 'data', 'NLPDs', and 'errors'.
            These fields contain the relevant subset of observations as a DataSet, an array of the negative log
            posterior densities of observations given the predictions, and an array of the natural-space difference
            between observations and prediction means, respectively.
        """

        if not (n_train is None) ^ (pct_train is None):
            raise ValueError('Exactly one of "n_train" and "pct_train" must be specified')

        if unit is not None:
            if not isinstance(unit, str):
                raise TypeError('Keyword "unit" must be a single string.')

        assert_in('Keyword "errors"', errors, ['natural', 'standardized', 'transformed'])

        seed = self.seed if seed is None else seed
        rg = np.random.default_rng(seed)

        df = self.data.wide

        n_entities = len(set(df.index)) if unit is None else len(set(df.set_index(unit).index))
        n_train = n_train if n_train is not None else np.floor(n_entities * pct_train).astype(int)
        if n_train <= 0:
            raise ValueError('Size of training set must be strictly greater than zero.')
        if n_train > n_entities:
            raise ValueError('Size of training set must be not exceed number of observations or entities in dataset.')

        # Build up a list of dataframes that make up the training set
        train_list = []

        if train_only is not None:
            # Move items that match `train_only` criteria to training set
            train_only_criteria = [df[dim] == level for dim, level in train_only.items()]
            train_only_idxs = pd.concat(train_only_criteria, axis=1).all(axis=1).index
            train_only_df = df.loc[train_only_idxs] if unit is None else df.loc[train_only_idxs].set_index(unit)
            n_train -= len(set(train_only_df.index))
            if n_train < 0:
                raise ValueError('Adding `train_only` observations exceeded specified size of training set')
            train_list.append(train_only_df)
            df = df.drop(index=train_only_idxs)

        # Group data by columns specified as "unit"
        if unit is not None:
            df = df.set_index(unit)
            remaining_entities = set(df.index)
            if len(train_list) > 1:
                # Ensure train_only didn't partially slice a unique entity
                train_only_entities = set(train_list[-1].index)
                if len(train_only_entities.intersection(remaining_entities)) > 0:
                    raise ValueError('Criteria in `train_only` partially sliced an entity specified by `unit`, which makes \
                                      interpretation of `n_train` ambiguous.')

        if n_train > len(df.index.unique()):
            raise ValueError('Specified size of training set exceeds number of unique combinations found in `dims`')

        if warm_start:
            # Add one random item from each categorical level to the training set

            if len(self.categorical_dims) > 0:
                # Filter out any observations not in the specified categorical levels
                level_combinations = list(product(*self.categorical_levels.values()))
                cat_grps = (df
                            .groupby(self.categorical_dims)
                            .filter(lambda grp: grp.name not in level_combinations)
                            .groupby(self.categorical_dims))

                if cat_grps.ngroups == 0:
                    raise ValueError(f'None of the combinations of categorical levels were found in data.\nCombinations:\n{level_combinations}')

                # Randomly select one item from each group
                warm_idxs = cat_grps.sample(1, random_state=seed).index
                if len(set(warm_idxs)) != len(warm_idxs):
                    warnings.warn('Duplicate entities specified by `unit` were selected during `warm_start`. This may lead to unexpected behavior.')
                n_train -= len(set(warm_idxs))
                if n_train < 0:
                    raise ValueError('Adding `warm_start` observations exceeded specified size of training set')
                train_list.append(df.loc[warm_idxs])
                df = df.drop(index=warm_idxs)

        # Move a random subset of the remaining items to the training set
        train_idxs = rg.choice(df.index.unique(), n_train, replace=False)
        for_train = df.loc[train_idxs]
        train_list.append(for_train)
        train_df = pd.concat(train_list).reset_index()
        test_df = df.drop(train_idxs).reset_index()

        categorical_dims = [dim for dim in self.categorical_dims if dim != self.out_col]

        specifications = dict(outputs=self.outputs, linear_dims=self.linear_dims, continuous_dims=self.continuous_dims,
                              continuous_levels=self.continuous_levels, continuous_coords=self.continuous_coords,
                              categorical_dims=categorical_dims, categorical_levels=self.categorical_levels,
                              additive=self.additive)

        train_specs = specifications | {
            'continuous_levels': {dim: [lvl for lvl in lvls if lvl in train_df[dim].values]
                                  for dim, lvls in self.continuous_levels.items()},
            'categorical_levels': {dim: [lvl for lvl in lvls if lvl in train_df[dim].values]
                                   for dim, lvls in self.categorical_levels.items()},
            'continuous_coords': {dim: {lvl: coord for lvl, coord in coords.items() if lvl in train_df[dim].values}
                                  for dim, coords in self.continuous_coords.items()}
        }

        test_specs = specifications | {
            'continuous_levels': {dim: [lvl for lvl in lvls if lvl in test_df[dim].values]
                                  for dim, lvls in self.continuous_levels.items()},
            'categorical_levels': {dim: [lvl for lvl in lvls if lvl in test_df[dim].values]
                                   for dim, lvls in self.categorical_levels.items()},
            'continuous_coords': {dim: {lvl: coord for lvl, coord in coords.items() if lvl in test_df[dim].values}
                                  for dim, coords in self.continuous_coords.items()}
        }

        dataset_specs = dict(outputs=self.data.outputs,
                             names_column=self.data.names_column,
                             values_column=self.data.values_column,
                             log_vars=self.data.log_vars,
                             logit_vars=self.data.logit_vars,
                             stdzr=self.data.stdzr)

        train_ds = DataSet(train_df, **dataset_specs)
        test_ds = DataSet(test_df, **dataset_specs)

        # Build and fit a new object of the current class (GP, GLM, etc) with the training set
        train_obj = self.__class__(train_ds, outputs=self.outputs, seed=seed)
        train_obj.specify_model(**train_specs)
        train_obj.filter_dims = self.filter_dims
        train_obj.build_model(**self.model_specs)
        train_obj.find_MAP(**MAP_kws)  # TODO: make more general to allow alternative inference approaches

        # Get in-sample prediction metrics
        train_X, train_y = train_obj.get_structured_data()
        train_predictions = train_obj.predict_points(train_X)
        train_nlpd = train_predictions.nlpd(train_y.values())
        train_error = {
            'natural': train_y.values() - train_predictions.μ,
            'transformed': train_y.t.values() - train_predictions.t.μ,
            'standardized': train_y.z.values() - train_predictions.z.μ,
        }[errors]

        if len(test_df.index.unique()) > 0:
            # If there's anything left for a testing set, build and fit a new object with the testing set
            test_obj = self.__class__(test_ds, outputs=self.outputs, seed=seed)

            # TODO: figure out why this was necessary and get rid of it
            categorical_dims = [dim for dim in self.categorical_dims if dim != self.out_col]
            test_specs['categorical_dims'] = categorical_dims
            train_specs['categorical_dims'] = categorical_dims
            test_obj.specify_model(**test_specs)
            test_obj.filter_dims = self.filter_dims

            # Get out-of-sample prediction metrics
            test_X, test_y = test_obj.get_structured_data()
            test_predictions = train_obj.predict_points(test_X)
            test_nlpd = test_predictions.nlpd(test_y.values())
            test_error = {
                'natural': test_y.values() - test_predictions.μ,
                'transformed': test_y.t.values() - test_predictions.t.μ,
                'standardized': test_y.z.values() - test_predictions.z.μ,
            }[errors]
        else:
            test_nlpd = np.nan
            test_error = np.nan

        result = {
            'train': {
                'data': train_ds,
                'NLPDs': train_nlpd,
                'errors': train_error},
            'test': {
                'data': test_ds,
                'NLPDs': test_nlpd,
                'errors': test_error}
        }

        return result

    ################################################################################
    # Plotting
    ################################################################################

    def get_conditional_prediction(self, **dim_values):
        """The conditional prediction at the given values of the specified dimensions over the remaining dimension(s).

        Conditioning the prediction on specific values of `m` dimensions can be thought of as taking a "slice" along the
        remaining `n` dimensions.

        Performs `(m+n)`-dimensional interpolation over the entire prediction grid for each of the mean and variance
        separately, then returns the interpolation evaluated at the specified values for the provided dimensions and the
        original values for the remaining dimensions.

        Parameters
        ----------
        dim_values
            Keyword arguments specifying value for each dimension at which to return the conditional prediction of the
            remaining dimensions.

        Returns
        -------
        conditional_grid: ParameterArray
            `n`-dimensional grid with `n` parameters (layers) at which the conditional prediction is evaluated
        conditional_prediction: UncertainParameterArray
            `n`-dimensional grid of predictions conditional on the given values of the `m` specified dimensions
        """

        self._check_has_prediction()
        all_dims = self.prediction_dims

        # All points along every axis (parrays)
        # Note that these may not all be the same length
        all_margins = {dim: vec.squeeze() for dim, vec in self.grid_vectors.items() if dim in self.prediction_dims}

        # The dimensions to be "kept" are the ones not listed in kwargs
        keep = set(all_dims) - set(dim_values.keys())
        kept_margins = [all_margins[dim] for dim in self.prediction_dims if dim in keep]

        # parray grid of original points along all "kept" dimensions
        conditional_grid = self.parray(**{array.names[0]: array.values() for array in np.meshgrid(*kept_margins)})
        # Add specified value for each remaining dimension at all points, then unravel
        xi_parray = conditional_grid.add_layers(
            **{dim: np.full(conditional_grid.shape, value) for dim, value in dim_values.items()}
        ).ravel()

        # Stack standardized points into (ordinary) tall array, ensuring dimensions are in the right order for the model
        xi_pts = np.column_stack([xi_parray[dim].z.values() for dim in self.dims if dim in xi_parray.names])

        # Interpolate the mean and variance of the predictions
        # Swapping the first two axes is necessary because grids were generated using meshgrid's default "ij" indexing
        # but interpn expects "xy" indexing
        μ_arr = np.swapaxes(self.predictions.μ, 0, 1)
        μi = interpn([all_margins[dim].z.values() for dim in self.dims if dim in self.prediction_dims], μ_arr, xi_pts)
        σ2_arr = np.swapaxes(self.predictions.σ2, 0, 1)
        σ2i = interpn([all_margins[dim].z.values() for dim in self.dims if dim in self.prediction_dims], σ2_arr, xi_pts)

        conditional_prediction = self.uparray(self.predictions.name, μ=μi, σ2=σ2i).reshape(*conditional_grid.shape)

        return conditional_grid.squeeze(), conditional_prediction.squeeze()
