import gpflow
from gpflow.ci_utils import ci_niter
import tensorflow_probability as tfp
from gumbi.utils.gp_utils import get_ℓ_prior
from gumbi.aggregation import DataSet
from gumbi.arrays import ParameterArray as parray
from gumbi.arrays import ParameterArray
from typing import Optional
import numpy as np
import tensorflow as tf
from gumbi.utils.misc import assert_is_subset
from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter, _cast_to_dtype
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, to_default_float, ops
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.covariances.dispatch import Kuf, Kuu
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from .base import Regressor


class LVMOGP(GPModel, InternalDataTrainingLossMixin):
    def __init__(
            self,
            data: OutputData,
            X_data: tf.Tensor,
            X_data_fn: tf.Tensor,
            H_data_mean: tf.Tensor,
            H_data_var: tf.Tensor,
            kernel: Kernel,
            num_inducing_variables: Optional[int] = None,
            inducing_variable=None,
            H_prior_mean=None,
            H_prior_var=None,
    ):
        """
        Initialise  LVMOGP object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D_out (output dimensions)
        :param X_data: observed inputs, size N (number of points) x D (input dimensions)
        :param H_data: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param H_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param H_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param H_prior_var: prior variance used in KL term of bound. By default 1.
        """
        num_data, num_latent_gps = X_data.shape
        num_fns, num_latent_dims = H_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)

        self.X_data = Parameter(X_data, trainable=False)
        self.X_data_fn = Parameter(X_data_fn, trainable=False)
        self.H_data_mean = Parameter(H_data_mean)
        self.H_data_var = Parameter(H_data_var, transform=positive())

        self.num_fns = num_fns
        self.num_latent_dims = num_latent_dims
        self.num_data = num_data
        self.output_dim = self.data.shape[-1]

        assert X_data.shape[0] == self.data.shape[0], "X mean and Y must be same size."
        assert H_data_mean.shape[0] == H_data_var.shape[0], "H mean and var should be the same length"

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-
            X_mean_tilde, X_var_tilde = self.fill_Hs()
            Z = tf.random.shuffle(X_mean_tilde)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if H_prior_mean is None:
            H_prior_mean = tf.zeros((self.num_fns, self.num_latent_dims), dtype=default_float())
        if H_prior_var is None:
            H_prior_var = tf.ones((self.num_fns, self.num_latent_dims))

        self.H_prior_mean = tf.convert_to_tensor(np.atleast_1d(H_prior_mean), dtype=default_float())
        self.H_prior_var = tf.convert_to_tensor(np.atleast_1d(H_prior_var), dtype=default_float())

        assert self.H_prior_mean.shape[0] == self.num_fns
        assert self.H_prior_mean.shape[1] == self.num_latent_dims
        assert self.H_prior_var.shape[0] == self.num_fns
        assert self.H_prior_var.shape[1] == self.num_latent_dims

    def fill_Hs(self):
        """append latent Hs to Xs by function number, to give X_tilde"""

        H_mean_vect = tf.reshape(tf.gather(_cast_to_dtype(self.H_data_mean, dtype=default_float()),
                                           _cast_to_dtype(self.X_data_fn, dtype=tf.int64)),
                                 [self.num_data, self.num_latent_dims])
        H_var_vect = tf.reshape(tf.gather(_cast_to_dtype(self.H_data_var, dtype=default_float()),
                                          _cast_to_dtype(self.X_data_fn, dtype=tf.int64)),
                                [self.num_data, self.num_latent_dims])

        return tf.concat([self.X_data, H_mean_vect], axis=1), \
               tf.concat([tf.ones(self.X_data.shape, dtype=default_float()) * 1e-5, H_var_vect], axis=1)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data = self.data
        mu, var = self.fill_Hs()
        pH = DiagonalGaussian(mu, var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 = tf.reduce_sum(expectation(pH, self.kernel))
        psi1 = expectation(pH, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pH, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        # tf.print(B)
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        # KL[q(x) || p(x)]
        dH_data_var = (
            self.H_data_var
            if self.H_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.H_data_var)
        )
        NQ = to_default_float(tf.size(self.H_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dH_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.H_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.H_data_mean - self.H_prior_mean) + dH_data_var) / self.H_prior_var
        )

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        # tf.print(bound)
        return bound

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        Y_data = self.data
        X_data = self.X_data
        X_mean_tilde, X_var_tilde = self.fill_Hs()
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_mean_tilde)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                    self.kernel(Xnew)
                    + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                    - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                    self.kernel(Xnew, full_cov=False)
                    + tf.reduce_sum(tf.square(tmp2), 0)
                    - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, 1])  # self.num_latent_gps

        return mean + self.mean_function(Xnew), var

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

    def predict_points(self, points, param=None, with_noise=True, lg10_Copies=None, **kwargs):
        """Make predictions at supplied points

        Parameters
        ----------
        points : ParameterArray
            1-D ParameterArray vector of coordinates for prediction, must have one layer per ``self.dims``
        param : str or list of str, optional
            Parameter for which to make predictions
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error
        lg10_Copies : float, optional
            Single value of lg10_Copies applied to all points
        **kwargs
            Additional keyword arguments passed to subclass-specific :meth:`predict` method

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a `uparray`
        """

        points = np.atleast_1d(points)
        assert points.ndim == 1
        assert set(self.dims) - set(['Parameter']) == set(points.names), \
            'All model dimensions must be present in "points" parray.'

        if 'Parameter' in self.categorical_dims:
            # Multiple parameters are possible, determine which ones to predict
            if param is None:
                # predict all parameters in model
                param = self.categorical_levels['Parameter']
            elif isinstance(param, list):
                assert_is_subset('Parameters', param, self.categorical_levels['Parameter'])
            elif isinstance(param, str):
                param = [param]
                assert_is_subset('Parameters', param, self.categorical_levels['Parameter'])
            else:
                raise ValueError('"param" must be list, string, or None')

            # Get model coordinates for each parameter to be predicted
            param_coords = [self.coregion_coords['Parameter'][p] for p in param]

            # Convert input points to tall array and tile once for each parameter, adding the respective coordinate
            tall_points = parray.vstack([points.add_layers(Parameter=coord)[:, None] for coord in param_coords])
        else:
            # If 'Parameter' is not in categorical_dims, it must be in filter_dims, and only one is possible
            param = self.filter_dims['Parameter']
            # Convert input points to tall array

            tall_points = points[:, None]

        points_array = np.hstack([tall_points[dim].z.values() for dim in self.dims])

        idx_c = [self.dims.index(dim) for dim in self.categorical_dims]
        if hasattr(self, 'lvmogps'):
            test = [self.model.H_data_mean[int(point)] for point in points_array[:, 2]]
            test2 = points_array[:, -1 * len(self.categorical_dims)]
            points_array = np.hstack([points_array[:, -1 * len(self.categorical_dims)].reshape(1, 1),
                                      np.array(
                                          [self.model.H_data_mean[int(point)] for point in points_array[:, 2]]).reshape(
                                          1, 2)])

        # Prediction means and variance as a list of numpy vectors
        pred_mean, pred_variance = self.predict(points_array, with_noise=with_noise, **kwargs)
        self.predictions_X = points

        if 'τ' in param:
            # Extract copy number from `points` if present, otherwise use the value provided in the function call
            lg10_Copies = points.get('lg10_Copies', lg10_Copies)
            if lg10_Copies is None and 'lg10_Copies' in self.filter_dims.keys():
                lg10_Copies = self.filter_dims['lg10_Copies']
            elif lg10_Copies is None:
                raise ValueError('Cannot predict τ without lg10_Copies')

            # Get standardized copy number
            if type(lg10_Copies) in [int, float, list, np.ndarray]:
                lg10_Copies = self.parray(lg10_Copies=lg10_Copies)
            lg10_Copies = lg10_Copies.z.values()
        else:
            lg10_Copies = None

        # Store predictions in appropriate structured array format
        if len(param) == 1:
            # Predicting one parameter, return an UncertainParameterArray
            self.predictions = self.uparray(param[0], pred_mean, pred_variance, lg10_Copies=lg10_Copies, stdzd=True)
        # else:
        #     # Predicting multiple parameters, return an MVUncertainParameterArray
        #     # First split prediction into UncertainParameterArrays
        #     uparrays = []
        #     for i, name in enumerate(param):
        #         idx = (tall_points['Parameter'].values() == param_coords[i]).squeeze()
        #         μ = pred_mean[idx]
        #         σ2 = pred_variance[idx]
        #         uparrays.append(self.uparray(name, μ, σ2, lg10_Copies=lg10_Copies, stdzd=True))
        #
        #     # Store predictions as MVUncertainParameterArray
        #     self.predictions = self.mvuparray(*uparrays, cor=cor)

        return self.predictions


class GP_gpflow(Regressor):
    r"""Gaussian Process surface learning and prediction.

    See Also
    --------
    :class:`Regressor`

    Notes
    -----
    This is the same as the GP class, expect it is implemented using GPflow rather than pymc3. The GP_gpflow
     class is built from a dataframe in the form of a :class:`DataSet` object. This is stored as
    :attr:`data`. The model inputs are constructed by filtering this dataframe, extracting column values, and
    converting these to numerical input coordinates. The main entry point will be :meth:`fit`, which parses the
    dimensions of the model with :meth:`specify_model`, extracts numerical input coordinates with
    :meth:`get_shaped_data`, compiles the Pymc3 model with :meth:`build_model`, and finally learns the
    hyperparameters with :meth:`find_MAP`.

    Dimensions fall into several categories:

    * Filter dimensions, those with only one level, are used to subset the dataframe but are not included as explicit
      inputs to the model. These are not specified explicitly, but rather any spatial or coregion dimension with only one
      level is treated as a filter dimension.
    * Spatial dimensions are treated as explicit coordinates and given a Radial Basis Function kernel

      * Linear dimensions (which must be a subset of `continuous_dims`) have an additional linear kernel.

    * Coregion dimensions imply a distinct but correlated output for each level

      * If more than one parameter is specified, ``'Parameter'`` is treated as a coregion dim.

    A non-additive model has the form:

    .. math::

        y &\sim \text{Normal} \left( \mu, \sigma \right) \\
        mu &\sim \mathcal{GP} \left( K \right) \\
        K &= \left( K^\text{spatial}+K^\text{lin} \right) K^\text{coreg}_\text{outputs} \prod_{n} K^\text{coreg}_{n} \\
        K^\text{spatial} &= \text{RBF} \left( \ell_{i}, \eta \right) \\
        K^\text{lin} &= \text{LIN} \left( c_{j}, \tau \right) \\
        K^\text{coreg} &= \text{Coreg} \left( \boldsymbol{W}, \kappa \right) \\
        \sigma &\sim \text{Exponential} \left( 1 \right) \\

    Where :math:`i` denotes a spatial dimension, :math:`j` denotes a linear dimension, and :math:`n` denotes a
    coregion dimension (excluding ``'Parameter'``). :math:`K^\text{spatial}` and :math:`K^\text{lin}` each consist of a
    joint kernel encompassing all spatial and linear dimensions, respectively, whereas :math:`K^\text{coreg}_{n}` is
    a distinct kernel for a given coregion dimension.

    The additive model has the form:

    .. math::

        mu &\sim \mathcal{GP}\left( K^\text{global} \right) + \sum_{n} \mathcal{GP}\left( K_{n} \right) \\
        K^\text{global} &= \left( K^\text{spatial}+K^\text{lin} \right) K^\text{coreg}_\text{outputs} \\
        K_{n} &= \left( K^\text{spatial}_{n}+K^\text{lin}_{n} \right) K^\text{coreg}_\text{outputs} K^\text{coreg}_{n} \\

    Note that, in the additive model, :math:`K^\text{spatial}_{n}` and :math:`K^\text{lin}_{n}` still consist of
    only the spatial and linear dimensions, respectively, but have unique priors corresponding to each coregion
    dimension. However, there is only one :math:`K^\text{coreg}_\text{outputs}` kernel.

    Internally, GC content is always on [0, 1], though it may be plotted on [0,100].

    Parameters
    ----------
   dataset : DataSet
        Data for fitting.
    outputs : str or list of str, default "r"
        Name(s) of parameter(s) to learn.
    seed : int
        Random seed

    Examples
    --------
    A GP object is created from a :class:`DataSet` and can be fit immediately with the default dimension
    configuration (regressing `r` with RBF + linear kernels for `BP` and `GC`):

    >>> from candas.learn import DataSet, GP
    >>> ps = DataSet.load('my_DataSet.pkl')
    >>> gp = GP_gpflow(ps).fit()

    Note that the last line is equivalent to

    >>> gp = GP_gpflow(ps)
    >>> gp.specify_model()
    >>> gp.build_model()
    >>> gp.find_MAP()

    The model can be specified with various spatial, linear, and coregion dimensions.
    `GC` and `BP` are always included in both ``continuous_dims`` and ``linear_dims``.

    >>> gp.specify_model(continuous_dims='lg10_Copies', linear_dims='lg10_Copies', categorical_dims='PrimerPair')
    >>> GP_gpflow(ps).fit(continuous_dims='lg10_Copies', linear_dims='lg10_Copies', categorical_dims='PrimerPair')  # equivalent

    After the model is fit, define a grid of points at which to make predictions. The result is a
    :class:`ParameterArray`:

    >>> gp.spatial_grid()
    >>> gp.grid_points
    ('GC', 'BP'): [(0.075     ,  10.) (0.08358586,  10.) (0.09217172,  10.) ...
     (0.90782828, 800.) (0.91641414, 800.) (0.925     , 800.)]

    Make predictions, returning an :class:`UncertainParameterArray`

    >>> gp.predict_grid()
    >>> gp.predictions
    r['?', '?2']: [[(0.70728056, 0.16073197) (0.70728172, 0.16073197)
                    (0.70728502, 0.16073197) ... (0.70727954, 0.16073197)
                    (0.7072809 , 0.16073197) (0.70728058, 0.16073197)]
                   ...
                   [(0.70749247, 0.1607318 ) (0.70773573, 0.16073116)
                    (0.70806603, 0.16072949) ... (0.70728449, 0.16073197)
                    (0.70728194, 0.16073197) (0.7072807 , 0.16073197)]]

    The `uparray` makes it easy to calculate standard statistics in natural or transformed/standardized space while
    maintaining the original shape of the array:

    >>> gp.predictions.z.dist.ppf(0.025)
    array([[-3.1887916 , -3.18878491, -3.18876601, ..., -3.18879744,
            -3.18878966, -3.18879146],
           ...,
           [-3.1875742 , -3.18617286, -3.18426272, ..., -3.18876906,
            -3.18878366, -3.18879081]])

     Finally, plot the results:

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.style.use(str(gmb.style.futura))
    >>> x_pa = gp.predictions_X
    >>> y_upa = gp.predictions
    >>> gmb.ParrayPlotter(x_pa, y_upa).plot()

    Plot a slice down the center of the prediction along each axis

    >>> x_pa, y_upa = gp.get_conditional_prediction(Y=88)
    >>>
    >>> ax = gmb.ParrayPlotter(x_pa, y_upa).plot()
    >>> ax.set_xticklabels([int(float(txt.get_text())*100) for txt in ax.get_xticklabels()]);

    Plot a slice down the center of the prediction along each axis

    >>> x_pa, y_upa = gp.get_conditional_prediction(X=0.5)
    >>>
    >>> ax = gmb.ParrayPlotter(x_pa, y_upa).plot()
    >>> ax.set_xticklabels([int(float(txt.get_text())*100) for txt in ax.get_xticklabels()]);


    Attributes
    ----------
    data : DataSet
        Data for fitting.
    outputs : list of str
        Name(s) of parameter(s) to learn.
    seed : int
        Random seed
    continuous_dims : list of str
        Columns of dataframe used as spatial dimensions
    linear_dims : list of str
        Subset of spatial dimensions to apply an additional linear kernel.
    continuous_levels : dict
        Values considered within each spatial column as ``{dim: [level1, level2]}``
    continuous_coords : dict
        Numerical coordinates of each spatial level within each spatial dimension as ``{dim: {level: coord}}``
    categorical_dims : list of str
        Columns of dataframe used as coregion dimensions
    categorical_levels : dict
        Values considered within each coregion column as ``{dim: [level1, level2]}``
    coregion_coords : dict
        Numerical coordinates of each coregion level within each coregion dimension as ``{dim: {level: coord}}``
    additive : bool
        Whether to treat coregion dimensions as additive or joint
    filter_dims : dict
        Dictionary of column-value pairs used to filter dataset before fitting
    X : array
        A 2D tall array of input coordinates.
    y : array
        A 1D vector of observations
    model : pymc3.model.Model
        Compiled pymc3 model
    gp_dict : dict
        Dictionary of model GP objects. Contains at least 'total'.
    """

    def __init__(self, dataset: DataSet, outputs=None, seed=2021):
        super(GP_gpflow, self).__init__(dataset, outputs, seed)

        self.model = None
        self.gp_dict = None
        self.MAP = None

        self.continuous_kernel = 'ExpQuad'
        self.heteroskedastic_inputs = False
        self.heteroskedastic_outputs = True
        self.sparse = False
        self.n_u = 100

        self.model_specs = {
            'seed': self.seed,
            'continuous_kernel': self.continuous_kernel,
            'heteroskedastic_inputs': self.heteroskedastic_inputs,
            'heteroskedastic_outputs': self.heteroskedastic_outputs,
            'sparse': self.sparse,
            'n_u': self.n_u,
        }

    ################################################################################
    # Model building and fitting
    ################################################################################

    def fit(self, outputs=None, linear_dims=None, continuous_dims=None, continuous_levels=None, continuous_coords=None,
            categorical_dims=None, categorical_levels=None, additive=False, seed=None, heteroskedastic_inputs=False,
            heteroskedastic_outputs=True, sparse=False, n_u=100, coregion_rank=None, **MAP_kwargs):
        """Fits a GP surface

        Parses inputs, compiles a gpflow model, then finds the MAP value for the hyperparameters. `{}_dims` arguments
        indicate the columns of the dataframe to be included in the model, with `{}_levels` indicating which values of
        those columns are to be included (``None`` implies all values).

        If ``additive==True``, the model is constructed as the sum of a global GP and a distinct GP for each coregion
        dimension. Each of these GPs, including the global GP, consists of an RBF+linear kernel multiplied by a
        coregion kernel for ``'Parameter'`` if necessary. Although the same spatial kernel structure is used for each
        GP in this model, unique priors are assigned to each distinct kernel. However, there is always only one
        coregion kernel for ``'Parameter'``. The kernel for each dimension-specific GP is further multiplied by a
        coregion kernel that provides an output for each level in that dimension.

        See Also
        --------
        :meth:`build_model`


        Parameters
        ----------
        outputs : str or list of str, default "r"
            Name(s) of parameter(s) to learn. If ``None``, :attr:`outputs` is used.
        linear_dims : str or list of str, optional
            Subset of spatial dimensions to apply an additional linear kernel. If ``None``, defaults to ``['BP','GC']``.
        continuous_dims : str or list of str, optional
            Columns of dataframe used as spatial dimensions.
        continuous_levels : str, list, or dict, optional
            Values considered within each spatial column as ``{dim: [level1, level2]}``.
        continuous_coords : list or dict, optional
            Numerical coordinates of each spatial level within each spatial dimension as ``{dim: {level: coord}}``.
        categorical_dims : str or list of str, optional
            Columns of dataframe used as coregion dimensions.
        categorical_levels : str, list, or dict, optional
            Values considered within each coregion column as ``{dim: [level1, level2]}``.
        additive : bool, default False
            Whether to treat categorical_dims as additive or joint (default).
        seed : int, optional.
            Random seed for model instantiation. If ``None``, :attr:`seed` is used.
        heteroskedastic_inputs: bool, default False
            Whether to allow heteroskedasticity along spatial dimensions (input-dependent noise)
        heteroskedastic_outputs: bool, default True
            Whether to allow heteroskedasticity between multiple Parameter outputs (output-dependent noise)


        Returns
        -------
        self : :class:`GP`
        """

        self.specify_model(outputs=outputs, linear_dims=linear_dims, continuous_dims=continuous_dims,
                           continuous_levels=continuous_levels, continuous_coords=continuous_coords,
                           categorical_dims=categorical_dims, categorical_levels=categorical_levels,
                           additive=additive, coregion_rank=coregion_rank)

        self.build_model(seed=seed,
                         heteroskedastic_inputs=heteroskedastic_inputs,
                         heteroskedastic_outputs=heteroskedastic_outputs,
                         sparse=sparse, n_u=n_u)

        self.train_model()

        return self

    # TODO: add full probabilistic model description to docstring
    def build_model(self, seed=None, heteroskedastic_inputs=False, heteroskedastic_outputs=True, sparse=False, n_u=100,
                    linear_cov_type='gpflow_linear'):
        r"""Compile a gpflow model for the GP.

        Each dimension in :attr:`continuous_dims` is combined in an ExpQuad kernel with a principled
        :math:`\text{InverseGamma}` prior for each lengthscale (as `suggested by Michael Betancourt`_) and a
        :math:`\text{Gamma}\left(2, 1\right)` prior for variance.

        .. _suggested by Michael Betancourt: https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#4_adding_an_informative_prior_for_the_length_scale

        Parameters
        ----------
        seed : int, optional.
            Random seed. If ``None``, :attr:`seed` is used.
        heteroskedastic_inputs: bool, default False
            Whether to allow heteroskedasticity along spatial dimensions (input-dependent noise)
        heteroskedastic_outputs: bool, default True
            Whether to allow heteroskedasticity between multiple Parameter outputs (output-dependent noise)

        Returns
        -------
        self : :class:`GP`
        """

        if heteroskedastic_inputs:
            raise NotImplementedError('Heteroskedasticity over inputs is not yet implemented.')

        seed = self.seed if seed is None else seed
        self.seed = seed
        self.heteroskedastic_inputs = heteroskedastic_inputs
        self.heteroskedastic_outputs = heteroskedastic_outputs
        self.sparse = sparse
        self.n_u = n_u

        n_l = len(self.linear_dims)
        n_s = len(self.continuous_dims)
        n_c = len(self.categorical_dims)
        n_p = len(self.outputs)

        self.model_specs = {
            'seed': seed,
            'heteroskedastic_inputs': heteroskedastic_inputs,
            'heteroskedastic_outputs': heteroskedastic_outputs,
            'sparse': sparse,
            'n_u': n_u,
        }

        self.linear_cov_type = linear_cov_type
        X, y = self.get_shaped_data(metric='mean')

        D_in = len(self.dims)
        assert X.shape[1] == D_in

        idx_l = [self.dims.index(dim) for dim in self.linear_dims]  # linear (can ignore for now)
        idx_s = [self.dims.index(dim) for dim in self.continuous_dims]  # spatial
        idx_c = [self.dims.index(dim) for dim in self.categorical_dims]  # coregionalisation
        idx_p = self.dims.index('Parameter') if 'Parameter' in self.dims else None

        if (linear_cov_type is None) & (self.linear_dims is not None):
            ValueError("Must specify the type of linear kernel out of 'gpflow_linear', "
                       "'linear+constant' and 'linear_offset'")

        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        seed = self.seed if seed is None else seed


        D_in = len(self.dims)
        assert X.shape[1] == D_in

        X_s = X[:, idx_s]

        ℓ_μ, ℓ_σ = [stat for stat in
                    np.array([get_ℓ_prior(dim) for dim in X_s.T]).T]  # this gets the values for the prior on X dims

        # these 3 functions define the different possible covariance matrices. One for X, one for linear (that can be ignored) and one for coreg
        def spatial_cov(suffix):

            lengthscales = tf.convert_to_tensor(ℓ_μ, dtype=default_float(), name='X_lengthscales')

            alphas = [mean ** 2 / ℓ_σ[i] + 2 for i, mean in enumerate(ℓ_μ)]
            betas = [mean * (mean ** 2 / ℓ_σ[i] + 1) for i, mean in enumerate(ℓ_μ)]
            k = gpflow.kernels.RBF(lengthscales=lengthscales, active_dims=idx_s)
            k.lengthscales.prior = tfp.distributions.InverseGamma(to_default_float(alphas), to_default_float(betas))
            k.variance.prior = tfp.distributions.Gamma(to_default_float(2), to_default_float(1))
            return k

        def linear_cov(suffix):
            "Must specify the type of linear kernel out of 'gpflow_linear', "
            "'linear+constant' and 'linear_offset'"

            var = tf.transpose(tf.convert_to_tensor([1.0] * n_l, dtype=default_float()))

            if self.linear_cov_type == 'gpflow_linear':
                k_l = gpflow.kernels.Linear(variance=var, active_dims=idx_l)
                k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))

                return k_l

            if self.linear_cov_type == 'linear+constant':
                k_l = gpflow.kernels.Linear(variance=var, active_dims=idx_l)
                k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))

                k_c = gpflow.kernels.Constant(variance=to_default_float(1.0), active_dims=idx_l)
                k_c.variance.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(10.0))

                return k_l + k_c

            if self.linear_cov_type == 'linear_offset':
                c = tf.transpose(tf.convert_to_tensor([1.0] * n_l, dtype=default_float()))
                k_l = kernels.Linear_offset(variance=var, offset=c, active_dims=idx_l)
                k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))
                k_l.offset.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(10.0))

                return k_l

        def coreg_cov(suffix, D_out, idx):

            coreg_k = gpflow.kernels.Coregion(output_dim=D_out, rank=self.coregion_rank,
                                              active_dims=[idx])  # TODO: change rank to be H_dims
            coreg_k.W.prior = tfp.distributions.Normal(to_default_float(0), to_default_float(3))
            coreg_k.kappa.prior = tfp.distributions.Gamma(to_default_float(2), to_default_float(1))
            return coreg_k

        # Define a "global" spatial kernel regardless of additive structure
        cov = spatial_cov('total')
        if n_l > 0:
            cov += linear_cov('total')

        # Construct a coregion kernel for each categorical_dims
        if n_c > 0 and not self.additive:  # I think I can probably ignore this additive parameter
            for dim, idx in zip(self.categorical_dims, idx_c):
                if dim == 'Parameter':
                    continue
                D_out = len(self.categorical_levels[dim])
                cov = cov * coreg_cov(dim, D_out, idx)

        # Coregion kernel for parameters, if necessary
        if 'Parameter' in self.categorical_dims:  # not sure what this is for
            D_out = len(self.categorical_levels['Parameter'])
            cov_param = coreg_cov('Parameter', D_out, idx_p)
            cov *= cov_param

        if sparse:
            Z = tf.random.shuffle(X)[:n_u]
            pm_gp = gpflow.models.SGPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                             tf.convert_to_tensor(y, dtype=default_float())), kernel=cov,
                                       inducing_variable=Z)
            gp_kws = {'approx': "FITC"}

        else:

            pm_gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                            tf.convert_to_tensor(y, dtype=default_float())), kernel=cov)

            gp_kws = {}

        pm_gp.likelihood.variance.prior = tfp.distributions.InverseGamma(to_default_float(2), to_default_float(1))
        gp_dict = {'total': pm_gp}
        # So I think this is just adding dimensions Not sure why
        # Construct a spatial+coregion kernel for each coregion_dim, then combine them additively
        if self.additive:
            gp_dict['global'] = gp_dict['total']
            for dim, idx in zip(self.categorical_dims, idx_c):
                if dim == 'Parameter':
                    continue

                # Spatial kernel specific to this dimension
                cov = spatial_cov(dim)
                # TODO: Decide if each additive dimension needs its own linear kernel
                if n_l > 0:
                    cov += linear_cov(dim)

                # Coregion kernel specific to this dimension
                D_out = len(self.categorical_levels[dim])
                cov *= coreg_cov(dim, D_out, idx)

                # Coregion kernel for parameters, if necessary
                if 'Parameter' in self.categorical_dims:
                    cov *= cov_param

                # Combine GPs
                gp_dict[dim] = pm_gp(cov_func=cov, **gp_kws)
                gp_dict['total'] += gp_dict[dim]

        # From https://docs.pymc.io/notebooks/GP-Marginal.html
        # OR a covariance function for the noise can be given
        # noise_l = pm.Gamma("noise_l", alpha=2, beta=2)
        # cov_func_noise = pm.gp.cov.Exponential(1, noise_l) + pm.gp.cov.WhiteNoise(sigma=0.1)
        # y_ = gp.marginal_likelihood("y", X=X, y=y, noise=cov_func_noise)

        # GP is heteroskedastic across Parameter outputs by default,
        # but homoskedastic across spatial dimensions
        # ? = pm.Exponential('?', lam=1)  # noise
        # noise = pm.gp.cov.WhiteNoise(sigma=?)
        # if heteroskedastic_inputs:
        #     raise NotImplementedError('Heteroskedasticity over inputs is not yet implemented')
        #     noise += spatial_cov('noise')
        # if heteroskedastic_outputs and 'Parameter' in self.categorical_dims:
        #     D_out = len(self.categorical_levels['Parameter'])
        #     noise *= coreg_cov('Parameter_noise', D_out, idx_p)
        #
        # if sparse:
        #     Xu = pm.gp.util.kmeans_inducing_points(n_u, X)  # inducing points
        #     if heteroskedastic_outputs:
        #         warnings.warn('Heteroskedasticity over outputs is not yet implemented for sparse GP. Reverting to scalar-valued noise.')
        #     _ = gp_dict['total'].marginal_likelihood('ml', X=X, Xu=Xu, y=y, noise=?)
        # else:
        #     _ = gp_dict['total'].marginal_likelihood('ml', X=X, y=y, noise=noise)

        self.gp_dict = gp_dict
        self.model = pm_gp
        return self

    def train_model(self, *args, **kwargs):
        """Finds maximum a posteriori value for hyperparameters in model using gpflow optimizer

        """
        assert self.model is not None

        maxiter = ci_niter(2000)
        res_LMC = gpflow.optimizers.Scipy().minimize(
            self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",
        )

    def predict(self, points_array, with_noise=True, additive_level='total', **kwargs):
        """Make predictions at supplied points using specified gp

        Parameters
        ----------
        param : str
        points : ParameterArray
            Tall ParameterArray vector of coordinates for prediction, must have one layer per ``self.dims``
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a `uparray`
        """

        # TODO: need to supply "given" dict for additive GP sublevel predictions
        if additive_level != 'total':
            raise NotImplementedError('Prediction for additive sublevels is not yet supported.')

        # Prediction means and variance as a numpy vector
        # predictions = self.gp_dict[additive_level].predict(points_array, point=self.MAP, diag=True,
        #                                                    pred_noise=with_noise, **kwargs)
        self.predictions = self.gp_dict[additive_level].predict_y(points_array)

        return self.predictions


class LVMOGP_GP(GP_gpflow):

    def __init__(self, lmc, lvmogp_latent_dims):
        self.lmc = lmc
        self.lvmogps = []
        self.gp_dict = {}

        self.lvmogp_latent_dims = lvmogp_latent_dims
        self.data = self.lmc.data
        self.stdzr = self.lmc.stdzr
        self.outputs = self.lmc.outputs
        self.seed = self.lmc.seed

        self.continuous_dims = self.lmc.continuous_dims
        self.linear_dims = self.lmc.linear_dims
        self.continuous_levels = self.lmc.continuous_levels
        self.continuous_coords = self.lmc.continuous_coords
        self.categorical_dims = self.lmc.categorical_dims
        self.categorical_levels = self.lmc.categorical_levels
        self.coregion_coords = self.lmc.coregion_coords
        self.filter_dims = self.lmc.filter_dims
        self.linear_cov_type = self.lmc.linear_cov_type
        self.additive = False

        self.X = self.lmc.X
        self.y = self.lmc.y

        self.predictions = None

        if lvmogp_latent_dims != self.lmc.coregion_rank:
            raise ValueError("LVMOGP latent dimensions and LMC rank are not equal")

    def fit(self, outputs=None, linear_dims=None, continuous_dims=None, continuous_levels=None, continuous_coords=None,
            categorical_dims=None, categorical_levels=None, additive=False, seed=None, heteroskedastic_inputs=False,
            heteroskedastic_outputs=True, sparse=False, n_u=100, **MAP_kwargs):

        self.build_model(seed=seed, n_u=n_u)
        self.train_model()

    def build_model(self, seed=None, heteroskedastic_inputs=False, heteroskedastic_outputs=True, sparse=False, n_u=100,
                    plot_BGPLVM=False, n_restarts=4):

        if 'Parameter' in self.dims:
            ordered_outputs = {k: v for k, v in sorted(self.coords['Parameter'].items(), key=lambda item: item[1])}
            y = np.hstack([self.y.z[param + '_z'].values() for param in ordered_outputs.keys()])
            X = np.atleast_2d(self.X)
            X = parray.vstack([X.add_layers(Parameter=coord) for coord in ordered_outputs.values()])
            X = np.atleast_2d(np.column_stack([X[dim].z.values().squeeze() for dim in self.dims]))

        else:
            y = self.y.z.drop('lg10_Copies_z').values().squeeze()
            X = np.atleast_2d(np.column_stack([self.X[dim].z.values().squeeze() for dim in self.dims]))

        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        idx_s = [self.dims.index(dim) for dim in self.continuous_dims]
        idx_c = [self.dims.index(dim) for dim in self.categorical_dims]
        X_s = X[:, idx_s]

        self.lmc.spatial_grid(resolution=50)
        means = []
        key = list(self.lmc.categorical_levels.keys())[0]
        for level in list(self.lmc.categorical_levels.values())[0]:
            predictions = self.lmc.predict_grid(param='r',
                                                categorical_levels={key: level})
            means.append(predictions['μ'])

        # fit GPLVM with restarts
        lmc_means = np.vstack(mean.flatten() for mean in means)

        if plot_BGPLVM:
            n_fun = len(lmc_means)
            fig, axs1 = plt.subplots(ncols=n_restarts, figsize=(n_restarts * 3, 3))
            f, ax = plt.subplots(nrows=3, ncols=n_restarts, figsize=(n_restarts * 3, 9))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2

        for j in range(n_restarts):
            try:
                lengthscales_latent_H = tf.convert_to_tensor(
                    [np.random.uniform(0.1, 1.0)] * int(self.lvmogp_latent_dims),
                    dtype=default_float(), name='H_lengthscales')
                kernel_H = gpflow.kernels.RBF(lengthscales=lengthscales_latent_H,
                                              active_dims=list(range(0, self.lvmogp_latent_dims)))
                kernel_H.variance.assign(np.random.uniform(0.1, 1.0))

                # priors
                kernel_H.lengthscales.prior = tfp.distributions.InverseGamma(to_default_float(2),
                                                                             to_default_float(1))
                kernel_H.variance.prior = tfp.distributions.InverseGamma(to_default_float(2),
                                                                         to_default_float(1))

                H_mean_init = ops.pca_reduce(tf.convert_to_tensor(lmc_means, dtype=default_float()),
                                             self.lvmogp_latent_dims)
                H_var_init = tf.ones((len(lmc_means), self.lvmogp_latent_dims), dtype=default_float())

                gplvm = gpflow.models.BayesianGPLVM(tf.convert_to_tensor(lmc_means, dtype=default_float()),
                                                    X_data_mean=H_mean_init,
                                                    X_data_var=H_var_init,
                                                    kernel=kernel_H,
                                                    num_inducing_variables=len(
                                                        list(self.lmc.categorical_levels.values())[0]),
                                                    X_prior_var=tf.ones(
                                                        (len(list(self.lmc.categorical_levels.values())[0]),
                                                         self.lvmogp_latent_dims))
                                                    )
                gplvm.likelihood.variance.prior = tfp.distributions.InverseGamma(to_default_float(2),
                                                                                 to_default_float(1))
                opt = gpflow.optimizers.Scipy()
                maxiter = ci_niter(2000)
                res = opt.minimize(
                    gplvm.training_loss,
                    method="BFGS",
                    variables=gplvm.trainable_variables,
                    options=dict(maxiter=maxiter),
                )

                #####
                if plot_BGPLVM:

                    lengths = gplvm.kernel.lengthscales.numpy()
                    dims = np.argsort(lengths)[-2:]

                    gplvm_X_mean = gplvm.X_data_mean.numpy()
                    gplvm_X_var = gplvm.X_data_var.numpy()
                    inducing = gplvm.inducing_variable.Z.numpy()
                    print(len(inducing))
                    X_pca = H_mean_init.numpy()

                    for i in range(n_fun):
                        axs1[j].scatter(X_pca[i, dims[0]], X_pca[i, dims[1]], label=i)
                        axs1[j].annotate(f'{i}', (X_pca[i, dims[0]], X_pca[i, dims[1]]))
                        axs1[j].set_title("PCA")
                        ax[0, j].scatter(gplvm_X_mean[i, dims[0]], gplvm_X_mean[i, dims[1]], label=i, color=colors[i], )
                        ax[0, j].annotate(f'{i}', (gplvm_X_mean[i, dims[0]], gplvm_X_mean[i, dims[1]]))
                        # ax[0, j].set_title(f'elbo {dic["elbos"][j]:.4f}')
                        ax[2, j].scatter(gplvm_X_mean[i, dims[0]], gplvm_X_mean[i, dims[1]], label=i, color=colors[i], )
                        ax[2, j].set_title("Bayesian GPLVM")
                        ax[2, j].set_xlabel(f'dim {dims[0]}')
                        ax[2, j].set_ylabel(f'dim {dims[1]}')
                        ax[0, j].set_xlabel(f'dim {dims[0]}')
                        ax[0, j].set_ylabel(f'dim {dims[1]}')
                        ax[1, j].scatter(gplvm_X_mean[i, dims[0]], gplvm_X_mean[i, dims[1]], color=colors[i], label=i)
                        circle1 = Ellipse((gplvm_X_mean[i, dims[0]], gplvm_X_mean[i, dims[1]]),
                                          1.95 * np.sqrt(gplvm_X_var[i, dims[0]]),
                                          1.95 * np.sqrt(gplvm_X_var[i, dims[1]]), color=colors[i], alpha=0.2, zorder=0)
                        ax[1, j].add_patch(circle1)
                        ax[1, j].annotate(f'{i}', (gplvm_X_mean[i, dims[0]], gplvm_X_mean[i, dims[1]]))

                        ax[2, j].scatter(inducing[:, dims[0]], inducing[:, dims[1]], marker='x', color='k')
                    fig.tight_layout()
                    f.tight_layout()

                # initalise the LVMOGP
                H_var = gplvm.X_data_var
                H_mean = tf.convert_to_tensor(gplvm.X_data_mean.numpy(), dtype=default_float())

                Z = tf.random.shuffle(X)[:n_u]
                inducing_variable = InducingPoints(Z)

                if len(self.linear_dims) == 0:

                    Ls = self.lmc.model.kernel.kernels[0].lengthscales.numpy().ravel().tolist() \
                         + gplvm.kernel.lengthscales.numpy().ravel().tolist()
                    lengthscales = tf.convert_to_tensor(Ls,
                                                        dtype=default_float(), name='H_lengthscales')

                else:

                    Ls = self.lmc.model.kernel.kernels[0].kernels[0].lengthscales.numpy().ravel().tolist() \
                         + gplvm.kernel.lengthscales.numpy().ravel().tolist()
                    lengthscales = tf.convert_to_tensor(Ls,
                                                        dtype=default_float(), name='H_lengthscales')

                k_s = gpflow.kernels.RBF(lengthscales=lengthscales)

                ℓ_μ, ℓ_σ = [stat for stat in
                            np.array(
                                [get_ℓ_prior(dim) for dim in X_s.T]).T]  # this gets the values for the prior on X dims

                alphas = [mean ** 2 / ℓ_σ[i] + 2 for i, mean in enumerate(ℓ_μ)] + [2] * self.lvmogp_latent_dims
                betas = [mean * (mean ** 2 / ℓ_σ[i] + 1) for i, mean in enumerate(ℓ_μ)] + [1] * self.lvmogp_latent_dims

                k_s.lengthscales.prior = tfp.distributions.InverseGamma(to_default_float(alphas),
                                                                        to_default_float(betas))
                k_s.variance.prior = tfp.distributions.InverseGamma(to_default_float(2), to_default_float(1))

                if len(self.linear_dims) > 0:

                    linear_variances = self.lmc.model.kernel.kernels[0].kernels[1].variance.numpy().ravel().tolist()
                    variances = linear_variances + [1e-3] * self.lvmogp_latent_dims

                    if self.linear_cov_type == 'gpflow_linear':
                        k_l = gpflow.kernels.Linear(variance=variances)
                        k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))

                    if self.linear_cov_type == 'linear+constant':
                        k_l = gpflow.kernels.Linear(variance=variances)
                        k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))

                        c = self.lmc.model.kernel.kernels[0].kernels[2].variance.numpy().ravel().tolist()
                        k_c = gpflow.kernels.Constant(variance=c)
                        k_c.variance.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(10.0))

                        k_l = k_l + k_c

                    if self.linear_cov_type == 'linear_offset':
                        linear_offsets = self.lmc.model.kernel.kernels[0].kernels[1].offset.numpy().ravel().tolist()
                        offsets = linear_offsets + [1e-3] * self.lvmogp_latent_dims
                        k_l = kernels.Linear_offset(variance=variances, offset=offsets)
                        k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))
                        k_l.offset.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(10.0))

                    else:
                        ValueError('linear kernel type must be specified')

                    kern = k_s + k_l

                else:
                    kern = k_s

                lvm = LVMOGP(data=y,
                             X_data=X[:, :(len(self.dims) - 1)],
                             X_data_fn=X[:, -1],
                             H_data_mean=H_mean,
                             H_data_var=H_var,
                             kernel=kern,
                             num_inducing_variables=n_u,
                             inducing_variable=None,
                             H_prior_mean=None,
                             H_prior_var=None, )
                lvm.likelihood.variance.prior = tfp.distributions.InverseGamma(to_default_float(2),
                                                                               to_default_float(1))
                # gp_dict = {'total': lvm}
                # self.gp_dict = gp_dict
                # self.model = lvm
                self.lvmogps.append(lvm)
            except:
                ValueError('LVMOGP failed to build')

        if plot_BGPLVM:
            plt.show()

    def train_model(self, *args, **kwargs):

        assert self.lvmogps is not None

        elbos = []
        for i, lvmogp in enumerate(self.lvmogps):
            try:
                opt = gpflow.optimizers.Scipy()
                maxiter = ci_niter(1000)
                res = opt.minimize(
                    lvmogp.training_loss,
                    method="BFGS",
                    variables=lvmogp.trainable_variables,
                    options=dict(maxiter=maxiter),
                )
                elbos.append(lvmogp.elbo())
            except:
                print(f'lvm {i} optimisation failed')
                elbos.append(float("nan"))
            # print(elbos[i])

        if np.isnan(elbos).all() == True:
            raise ValueError('all LVMOGPs failed to optimise')
            print('all LVMOGPs failed to optimise')

        # select model with largest elbo

        index = np.nanargmax(elbos)
        self.model = self.lvmogps[index]
        self.gp_dict['total'] = self.model
