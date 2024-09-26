import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from gumbi.aggregation import DataSet
from gumbi.arrays import (  # noqa: F401
    MVUncertainParameterArray,
    ParameterArray,
    UncertainParameterArray,
)
from gumbi.utils.gp_utils import get_ls_prior
from gumbi.utils.misc import assert_in

from ..base import Regressor

__all__ = ["PymcGP"]


class PymcGP(Regressor):
    r"""Gaussian Process surface learning and prediction.

    See Also
    --------
    :class:`Regressor`

    Notes
    -----
    The GP class is built from a dataframe in the form of a :class:`DataSet`
    object. The output(s) are taken from :attr:`DataSet.outputs` and the
    corresponding column in the tidy data frame taken from
    :attr:`DataSet.names_column`. This column will be generically referred to as
    the `output_column` in this documentation, but can take any value specifed
    when the :class:`DataSet` is constructed. The model inputs are constructed
    by filtering this dataframe, extracting column values, and converting these
    to numerical input coordinates. The main entry point will be :meth:`fit`,
    which parses the dimensions of the model with :meth:`specify_model`,
    extracts numerical input coordinates with :meth:`get_structured_data`,
    compiles the Pymc model with :meth:`build_model`, and finally learns the
    hyperparameters with :meth:`find_MAP`.

    Dimensions fall into several categories:

    * Filter dimensions, those with only one level, are used to subset the
      dataframe but are not included as explicit inputs to the model. These are
      not specified explicitly, but rather any continuous or categorical
      dimension with only one level is treated as a filter dimension.

    * Continuous dimensions are treated as explicit coordinates and given a
      Radial Basis Function kernel

      * Linear dimensions (which must be a subset of `continuous_dims`) have an
        additional linear kernel.

    * Coregion dimensions imply a distinct but correlated output for each level

      * If more than one output is specified, the `output_column` is treated as
        a categorical dim.

    A non-additive model has the form:

    .. math::

        y &\sim \text{Normal} \left( \mu, \sigma \right) \\ mu &\sim
        \mathcal{GP} \left( K \right) \\ K &= \left( K^\text{cont}+K^\text{lin}
        \right) K^\text{coreg}_\text{outputs} \prod_{n} K^\text{coreg}_{n} \\
        K^\text{cont} &= \text{RBF} \left( \ell_{i}, \eta \right) \\
        K^\text{lin} &= \text{LIN} \left( c_{j}, \tau \right) \\ K^\text{coreg}
        &= \text{Coreg} \left( \boldsymbol{W}, \kappa \right) \\ \sigma &\sim
        \text{Exponential} \left( 1 \right) \\

    Where :math:`i` denotes a continuous dimension, :math:`j` denotes a linear
    dimension, and :math:`n` denotes a categorical dimension (excluding the
    `output_column`). :math:`K^\text{cont}` and :math:`K^\text{lin}` each
    consist of a joint kernel encompassing all continuous and linear dimensions,
    respectively, whereas :math:`K^\text{coreg}_{n}` is a distinct kernel for a
    given categorical dimension.

    The additive model has the form:

    .. math::

        mu &\sim \mathcal{GP}\left( K^\text{global} \right) + \sum_{n}
        \mathcal{GP}\left( K_{n} \right) \\ K^\text{global} &= \left(
        K^\text{cont}+K^\text{lin} \right) K^\text{coreg}_\text{outputs} \\
        K_{n} &= \left( K^\text{cont}_{n}+K^\text{lin}_{n} \right)
        K^\text{coreg}_\text{outputs} K^\text{coreg}_{n} \\

    Note that, in the additive model, :math:`K^\text{cont}_{n}` and
    :math:`K^\text{lin}_{n}` still consist of only the continuous and linear
    dimensions, respectively, but have unique priors corresponding to each
    categorical dimension. However, there is only one
    :math:`K^\text{coreg}_\text{outputs}` kernel.

    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    outputs : str or list of str, default None
        Name(s) of output(s) to learn. If ``None``, uses all values from
        ``outputs`` attribute of *dataset*.
    seed : int
        Random seed

    Examples
    --------
    A GP object is created from a :class:`DataSet` and can be fit immediately
    with the default dimension configuration (regressing `r` with RBF + linear
    kernels for `X` and `Y`):

    >>> import gumbi as gmb
    >>> df = pd.read_pickle(gmb.data.example_dataset)
    >>> outputs=['a', 'b', 'c', 'd', 'e', 'f']
    >>> ds = gmb.DataSet(df, outputs=outputs, log_vars=['Y', 'b', 'c', 'd', 'f'], logit_vars=['X', 'e'])
    >>> gp = gmb.GP(ds, outputs='d').fit()

    Note that last line is equivalent to

    >>> gp = gmb.GP(ds, outputs='d')
    >>> gp.specify_model()
    >>> gp.build_model()
    >>> gp.find_MAP()

    The model can be specified with various continuous, linear, and categorical
    dimensions. `X` and `Y` are always included in both ``continuous_dims`` and
    ``linear_dims``.

    >>> gp.specify_model(continuous_dims='lg10_Z', linear_dims='lg10_Z', categorical_dims='Pair')
    >>> gmb.GP(ds).fit(continuous_dims='lg10_Z', linear_dims='lg10_Z', categorical_dims='Pair')  # equivalent

    After the model is fit, define a grid of points at which to make
    predictions. The result is a :class:`ParameterArray`:

    >>> gp.prepare_grid()
    >>> gp.grid_points
    ('X', 'Y'): [(0.075     ,  10.) (0.08358586,  10.) (0.09217172,  10.) ...
     (0.90782828, 800.) (0.91641414, 800.) (0.925     , 800.)]

    Make predictions, returning an :class:`UncertainParameterArray`

    >>> gp.predict_grid()
    >>> gp.predictions
    d['μ', 'σ2']: [[(0.70728056, 0.16073197) (0.70728172, 0.16073197)
                    (0.70728502, 0.16073197) ... (0.70727954, 0.16073197)
                    (0.7072809 , 0.16073197) (0.70728058, 0.16073197)]
                   ...
                   [(0.70749247, 0.1607318 ) (0.70773573, 0.16073116)
                    (0.70806603, 0.16072949) ... (0.70728449, 0.16073197)
                    (0.70728194, 0.16073197) (0.7072807 , 0.16073197)]]

    The `uparray` makes it easy to calculate standard statistics in natural or
    transformed/standardized space while maintaining the original shape of the
    array:

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
    dataset : DataSet
        Data for fitting.
    outputs : list of str
        Name(s) of output(s) to learn.
    seed : int
        Random seed
    continuous_dims : list of str
        Columns of dataframe used as continuous dimensions
    linear_dims : list of str
        Subset of continuous dimensions to apply an additional linear kernel.
    continuous_levels : dict
        Values considered within each continuous column as ``{dim: [level1,
        level2]}``
    continuous_coords : dict
        Numerical coordinates of each continuous level within each continuous
        dimension as ``{dim: {level: coord}}``
    categorical_dims : list of str
        Columns of dataframe used as categorical dimensions
    categorical_levels : dict
        Values considered within each categorical column as ``{dim: [level1,
        level2]}``
    categorical_coords : dict
        Numerical coordinates of each categorical level within each categorical
        dimension as ``{dim: {level: coord}}``
    additive : bool
        Whether to treat categorical dimensions as additive or joint
    filter_dims : dict
        Dictionary of column-value pairs used to filter dataset before fitting
    X : array
        A 2D tall array of input coordinates.
    y : array
        A 1D vector of observations
    model : pymc.model.Model
        Compiled pymc model
    gp_dict : dict
        Dictionary of model GP objects. Contains at least 'total'.
    """

    def __init__(self, dataset: DataSet, outputs=None, seed=2021):
        super(PymcGP, self).__init__(dataset, outputs, seed)

        self.model = None
        self.gp_dict = None
        self.MAP = None
        self.trace = None

        self.continuous_kernel = "ExpQuad"
        self.heteroskedastic_inputs = False
        self.heteroskedastic_outputs = True
        self.sparse = False
        self.latent = False
        self.n_u = 100

        self.model_specs = {
            "seed": self.seed,
            "continuous_kernel": self.continuous_kernel,
            "heteroskedastic_inputs": self.heteroskedastic_inputs,
            "heteroskedastic_outputs": self.heteroskedastic_outputs,
            "sparse": self.sparse,
            "n_u": self.n_u,
        }

    ################################################################################
    # Model building and fitting
    ################################################################################

    def fit(
        self,
        outputs=None,
        linear_dims=None,
        continuous_dims=None,
        continuous_levels=None,
        continuous_coords=None,
        categorical_dims=None,
        categorical_levels=None,
        additive=False,
        seed=None,
        continuous_kernel="ExpQuad",
        period=None,
        heteroskedastic_inputs=False,
        heteroskedastic_outputs=True,
        sparse=False,
        n_u=100,
        ARD=True,
        ls_bounds=None,
        mass=0.98,
        spec_kwargs=None,
        build_kwargs=None,
        MAP_kwargs=None,
    ):
        """Fits a GP surface

        Parses inputs, compiles a Pymc model, then finds the MAP value for the
        hyperparameters. `{}_dims` arguments indicate the columns of the
        dataframe to be included in the model, with `{}_levels` indicating which
        values of those columns are to be included (``None`` implies all
        values).

        If ``additive==True``, the model is constructed as the sum of a global
        GP and a distinct GP for each categorical dimension. Each of these GPs,
        including the global GP, consists of an RBF+linear kernel multiplied by
        a coregion kernel for the `output_column` if necessary. Although the
        same continuous kernel structure is used for each GP in this model,
        unique priors are assigned to each distinct kernel. However, there is
        always only one coregion kernel for the `output_column`. The kernel for
        each dimension-specific GP is further multiplied by a coregion kernel
        that provides an output for each level in that dimension.

        See Also
        --------
        :meth:`build_model`

        Parameters
        ----------
        outputs : str or list of str, default None
            Name(s) of output(s) to learn. If ``None``, :attr:`outputs` is used.
        linear_dims : str or list of str, optional
            Subset of continuous dimensions to apply an additional linear
            kernel. If ``None``, defaults to ``['Y','X']``.
        continuous_dims : str or list of str, optional
            Columns of dataframe used as continuous dimensions.
        continuous_levels : str, list, or dict, optional
            Values considered within each continuous column as ``{dim: [level1,
            level2]}``.
        continuous_coords : list or dict, optional
            Numerical coordinates of each continuous level within each
            continuous dimension as ``{dim: {level: coord}}``.
        categorical_dims : str or list of str, optional
            Columns of dataframe used as categorical dimensions.
        categorical_levels : str, list, or dict, optional
            Values considered within each categorical column as ``{dim: [level1,
            level2]}``.
        additive : bool, default False
            Whether to treat categorical_dims as additive or joint (default).
        seed : int, optional.
            Random seed for model instantiation. If ``None``, :attr:`seed` is
            used.
        continuous_kernel : {'ExpQuad', 'Matern32', 'Matern52', 'Exponential',
        or 'Cosine'}
            Covariance function to use for continuous dimensions. See `pymc
            docs`_ for more details.
        period : ParameterArray, optional
            A single parray of length 1 with one layer for each
            `continuous_dims` by name containing the period of the kernel, if
            periodic-like kernel is used.
        heteroskedastic_inputs: bool, default False
            Whether to allow heteroskedasticity along continuous dimensions
            (input-dependent noise)
        heteroskedastic_outputs: bool, default True
            Whether to allow heteroskedasticity between multiple outputs
            (output-dependent noise). `Not yet implemented`
        sparse: bool, default False
            Whether to use a `sparse approximation`_ to the GP.
        n_u: int, default 100
            Number of inducing points to use for the sparse approximation, if
            required.
        ARD: bool, default True
            Whether to use "Automatic Relevance Determination" in the continuous
            kernel. If _True_, each continuous dimension receives its own
            lengthscale; otherwise a single lengthscale is used for all
            continuous dimensions.

        **MAP_kwargs
            Additional keyword arguments passed to :func:`pm.find_MAP`.

        Returns
        -------
        self : :class:`GP`
        """

        self.specify_model(
            outputs=outputs,
            linear_dims=linear_dims,
            continuous_dims=continuous_dims,
            continuous_levels=continuous_levels,
            continuous_coords=continuous_coords,
            categorical_dims=categorical_dims,
            categorical_levels=categorical_levels,
            additive=additive,
            **(spec_kwargs or {}),
        )

        self.build_model(
            seed=seed,
            continuous_kernel=continuous_kernel,
            period=period,
            heteroskedastic_inputs=heteroskedastic_inputs,
            heteroskedastic_outputs=heteroskedastic_outputs,
            sparse=sparse,
            n_u=n_u,
            ARD=ARD,
            ls_bounds=ls_bounds,
            mass=mass,
            **(build_kwargs or {}),
        )

        self.find_MAP(**(MAP_kwargs or {}))

        return self

    def _make_continuous_cov(
        self,
        continuous_cov_func,
        D_in,
        idx_s,
        n_s,
        ls_params,
        ARD=True,
        period=None,
        **kernel_kwargs,
    ):
        shape = n_s if ARD else 1

        if period is not None:
            zperiods = [period.z[dim + "_z"].values() for dim in self.continuous_dims]
            kernel_kwargs["period"] = np.array(zperiods).squeeze()

        def continuous_cov(suffix):
            ls = pm.InverseGamma(f"ls_{suffix}", **ls_params, shape=shape)
            # ls = pm.Gamma(f"ls_{suffix}", alpha=2, beta=1, shape=shape)
            η = pm.Gamma(f"η_{suffix}", alpha=2, beta=1)
            cov = η**2 * continuous_cov_func(input_dim=D_in, active_dims=idx_s, ls=ls, **kernel_kwargs)

            return cov

        return continuous_cov

    def _make_periodic_cov(
        self,
        continuous_cov_func,
        D_in,
        idx_s,
        n_s,
        ls_params,
        period,
        ARD=True,
        **kernel_kwargs,
    ):
        continuous_kernel_factory = self._make_continuous_cov(
            continuous_cov_func, 2, None, n_s, ls_params, ARD=ARD, **kernel_kwargs
        )

        zperiods = [period.z[dim + "_z"].values() for dim in self.continuous_dims]
        zperiods = np.array(zperiods).squeeze() if len(zperiods) > 1 else zperiods[0]

        def mapping(x, zperiods):
            c = 2.0 * np.pi * (1.0 / zperiods)
            u = pt.concatenate((pt.sin(c * x), pt.cos(c * x)), 1)
            return u

        def periodic_cov(suffix):
            cov_func = continuous_kernel_factory(suffix)
            cov = pm.gp.cov.WarpedInput(
                input_dim=D_in, active_dims=idx_s, cov_func=cov_func, warp_func=mapping, args=(zperiods,)
            )

            return cov

        return periodic_cov

    def _make_linear_cov(self, D_in, idx_l, n_l):
        def linear_cov(suffix):
            c = pm.Normal(f"c_{suffix}", mu=0, sigma=10, shape=n_l)
            τ = pm.HalfNormal(f"τ_{suffix}", sigma=10)
            return τ * pm.gp.cov.Linear(input_dim=D_in, c=c, active_dims=idx_l)

        return linear_cov

    def _make_coreg_cov(self, D_in, seed):
        def coreg_cov(suffix, D_out, idx):
            initval = np.random.default_rng(seed).standard_normal(size=(D_out, 2))
            W = pm.Normal(f"W_{suffix}", mu=0, sigma=3, shape=(D_out, 2), initval=initval)
            κ = pm.Gamma(f"κ_{suffix}", alpha=1.5, beta=1, shape=(D_out,))
            return pm.gp.cov.Coregion(input_dim=D_in, active_dims=[idx], kappa=κ, W=W)

        return coreg_cov

    # TODO: add full probabilistic model description to docstring TODO: allow dimension-specific continuous kernel
    # specification TODO: allow single multi-dimensional continuous kernel rather than independent kernels per dimension
    def build_model(
        self,
        seed=None,
        continuous_kernel="ExpQuad",
        period=None,
        heteroskedastic_inputs=False,
        heteroskedastic_outputs=True,
        sparse=False,
        n_u=100,
        ARD=True,
        ls_bounds=None,
        mass=0.98,
    ):
        r"""Compile a marginalized pymc model for the GP.

        Each dimension in :attr:`continuous_dims` is combined in an ExpQuad kernel with a principled
        :math:`\text{InverseGamma}` prior for each lengthscale (as `suggested by Michael Betancourt`_) and a
        :math:`\text{Gamma}\left(2, 1\right)` prior for variance.

        .. _suggested by Michael Betancourt: https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#4_adding_an_informative_prior_for_the_length_scale  # noqa: E501
        .. _pymc docs: https://docs.pymc.io/en/v3/api/gp/cov.html
        .. _sparse approximation: https://docs.pymc.io/en/v3/pymc-examples/examples/gaussian_processes/GP-SparseApprox.html  # noqa: E501

        Parameters
        ----------
        seed : int, optional.
            Random seed. If ``None``, :attr:`seed` is used.
        continuous_kernel : {'ExpQuad', 'Matern32', 'Matern52', 'Exponential', 'Cosine', or 'Periodic'}
            Covariance function to use for continuous dimensions. See `pymc docs`_ for more details.
        period : ParameterArray, optional
            A single parray of length 1 with one layer for each `continuous_dims` by name containing the period of the kernel, if periodic-like kernel is used.
        heteroskedastic_inputs: bool, default False
            Whether to allow heteroskedasticity along continuous dimensions (input-dependent noise).
        heteroskedastic_outputs: bool, default True
            Whether to allow heteroskedasticity between multiple outputs (output-dependent noise). `Not yet
            implemented`.
        sparse: bool, default False
            Whether to use a `sparse approximation`_ to the GP.
        n_u: int, default 100
            Number of inducing points to use for the sparse approximation, if required.
        ARD: bool, default True
            Whether to use "Automatic Relevance Determination" in the continuous kernel. If _True_, each continuous
            dimension receives its own lengthscale; otherwise a single lengthscale is used for all continuous
            dimensions.

        Returns
        -------
        self : :class:`GP`
        """

        if heteroskedastic_inputs:
            raise NotImplementedError("Heteroskedasticity over inputs is not yet implemented.")

        X, y = self.get_shaped_data("mean")
        D_in = len(self.dims)
        assert X.shape[1] == D_in

        seed = self.seed if seed is None else seed
        self.seed = seed
        self.continuous_kernel = continuous_kernel
        self.heteroskedastic_inputs = heteroskedastic_inputs
        self.heteroskedastic_outputs = heteroskedastic_outputs
        self.sparse = sparse
        self.n_u = n_u
        self.latent = False

        self.model_specs = {
            "seed": seed,
            "continuous_kernel": continuous_kernel,
            "heteroskedastic_inputs": heteroskedastic_inputs,
            "heteroskedastic_outputs": heteroskedastic_outputs,
            "sparse": sparse,
            "n_u": n_u,
        }
        gp_dict = self._construct_kernels(
            X,
            continuous_kernel,
            seed,
            sparse,
            latent=False,
            ARD=ARD,
            ls_bounds=ls_bounds,
            mass=mass,
            period=period,
        )

        with self.model:
            # From https://docs.pymc.io/notebooks/GP-Marginal.html OR a covariance function for the noise can be given
            # noise_l = pm.Gamma("noise_l", alpha=2, beta=2) cov_func_noise = pm.gp.cov.Exponential(1, noise_l) +
            # pm.gp.cov.WhiteNoise(sigma=0.1) y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=cov_func_noise)

            # GP is heteroskedastic across outputs by default, but homoskedastic across continuous dimensions
            σ = pm.Exponential("σ", lam=1)
            noise = pm.gp.cov.WhiteNoise(sigma=σ)
            if heteroskedastic_inputs:
                raise NotImplementedError("Heteroskedasticity over inputs is not yet implemented")
                # noise += continuous_cov('noise')
            if heteroskedastic_outputs and self.out_col in self.categorical_dims:
                D_out = len(self.categorical_levels[self.out_col])
                coreg_cov = self._make_coreg_cov(D_in, seed)
                idx_p = self._get_dim_indexes()["p"]
                noise *= coreg_cov("Output_noise", D_out, idx_p)

            if sparse:
                Xu = pm.gp.util.kmeans_inducing_points(n_u, X)
                if heteroskedastic_outputs:
                    warnings.warn(
                        "Heteroskedasticity over outputs is not yet implemented for sparse GP. \
                            Reverting to scalar-valued noise."
                    )
                _ = gp_dict["total"].marginal_likelihood("ml", X=X, Xu=Xu, y=y, sigma=σ)
            else:
                _ = gp_dict["total"].marginal_likelihood("ml", X=X, y=y, sigma=noise)

        # self.gp_dict = gp_dict
        return self

    def _choose_implementation(self, sparse=False, latent=False):
        if sparse and latent:
            raise NotImplementedError("Sparse Latent GPs are not yet implemented.")

        if sparse:
            pm_gp = pm.gp.MarginalSparse
            gp_kws = {"approx": "FITC"}
        elif latent:
            pm_gp = pm.gp.Latent
            gp_kws = {}
        else:
            pm_gp = pm.gp.Marginal
            gp_kws = {}

        def implementation(*args, **kwargs):
            return pm_gp(*args, **{**kwargs, **gp_kws})  # Fix once Python >= 3.9

        return implementation

    def _get_dim_counts(self):
        dim_counts = {
            "l": len(self.linear_dims),
            "s": len(self.continuous_dims),
            "c": len(self.categorical_dims),
            "p": len(self.outputs),
        }

        return dim_counts

    def _get_dim_indexes(self):
        dim_indexes = {
            "l": [self.dims.index(dim) for dim in self.linear_dims],
            "s": [self.dims.index(dim) for dim in self.continuous_dims],
            "c": [self.dims.index(dim) for dim in self.categorical_dims],
            "p": self.dims.index(self.out_col) if self.out_col in self.dims else None,
        }

        return dim_indexes

    # def _prepare_lengthscales(self, X, ARD=False):
    #     X_s = X[:, self._get_dim_indexes()["s"]]

    #     ls_μ, ls_σ = [stat for stat in np.array([get_ls_prior(dim) for dim in X_s.T]).T]
    #     return ls_μ, ls_σ

    def _prepare_lengthscales(self, X, *, ARD, ls_bounds=None, mass=0.98):
        X_s = X[:, self._get_dim_indexes()["s"]]
        
        if ls_bounds is not None:
            zbounds = [
                [b if not np.isnan(b) else None for b in ls_bounds[dim].z.values().squeeze()]
                # if dim in ls_bounds.names else (None, None)
                for dim in self.continuous_dims
                if dim in ls_bounds.names
            ]
            lower, upper = list(zip(*zbounds))
            if not ARD and len(lower) != 1 or len(upper) != 1:
                raise ValueError("Bounds must be specified for only a single dimension if ARD is False")
        else:
            lower, upper = None, None
            
        ls_params = get_ls_prior(X_s, ARD=ARD, lower=lower, upper=upper, mass=mass)

        # ls_μ, ls_σ = [stat for stat in np.array([get_ls_prior(dim) for dim in X_s.T]).T]
        # return ls_μ, ls_σ
        return ls_params

    def _construct_kernels(
        self,
        X,
        continuous_kernel,
        seed,
        sparse,
        latent,
        ARD=True,
        ls_bounds=None,
        mass=0.98,
        period=None,
    ):
        continuous_kernels = [
            "ExpQuad",
            # "RatQuad",
            "Matern12",
            "Matern32",
            "Matern52",
            "Exponential",
            "Periodic",
        ]
        continuous_kernels += [kernel + "+Periodic" for kernel in continuous_kernels if kernel != "Periodic"]
        assert_in("Continuous kernel", continuous_kernel, continuous_kernels)

        if continuous_kernel.endswith("+Periodic"):
            continuous_kernel = continuous_kernel.removesuffix("+Periodic")
            if period is None:
                raise ValueError("Period must be specified for periodic kernel")
            kernel_factory = self._make_periodic_cov
        else:
            kernel_factory = self._make_continuous_cov

        continuous_cov_func = getattr(pm.gp.cov, continuous_kernel)

        D_in = len(self.dims)

        ns = self._get_dim_counts()
        idxs = self._get_dim_indexes()
        ls_params = self._prepare_lengthscales(X, ARD=ARD, ls_bounds=ls_bounds, mass=mass)

        continuous_cov = kernel_factory(
            continuous_cov_func,
            D_in,
            idxs["s"],
            ns["s"],
            ls_params,
            ARD=ARD,
            period=period,
        )
        linear_cov = self._make_linear_cov(D_in, idxs["l"], ns["l"])
        coreg_cov = self._make_coreg_cov(D_in, seed)

        pm_gp = self._choose_implementation(sparse=sparse, latent=latent)

        with pm.Model() as self.model:
            # μ = pm.Normal('μ', mu=0, sigma=10) β = pm.Normal('β', mu=0, sigma=10, shape=n_l) lin_mean =
            # pm.gp.mean.Linear(coeffs=[β[i] if i in idx_l else 0 for i in range(D_in), intercept=μ)

            # Define a "global" continuous kernel regardless of additive structure
            cov = continuous_cov("total")
            if ns["l"] > 0:
                cov += linear_cov("total")

            # Construct a coregion kernel for each categorical_dims
            if ns["c"] > 0 and not self.additive:
                for dim, idx in zip(self.categorical_dims, idxs["c"]):
                    if dim == self.out_col:
                        continue
                    D_out = len(self.categorical_levels[dim])
                    cov *= coreg_cov(dim, D_out, idx)

            # Coregion kernel for parameters, if necessary
            if self.out_col in self.categorical_dims:
                D_out = len(self.categorical_levels[self.out_col])
                cov_param = coreg_cov(self.out_col, D_out, idxs["p"])
                cov *= cov_param

            gp_dict = {"total": pm_gp(cov_func=cov)}

            # Construct a continuous+coregion kernel for each categorical_dim, then combine them additively
            if self.additive:
                gp_dict["global"] = gp_dict["total"]
                for dim, idx in zip(self.categorical_dims, idxs["c"]):
                    if dim == self.out_col:
                        continue

                    # Continuous kernel specific to this dimension
                    cov = continuous_cov(dim)
                    # TODO: Decide if each additive dimension needs its own linear kernel
                    if ns["l"] > 0:
                        cov += linear_cov(dim)

                    # Coregion kernel specific to this dimension
                    D_out = len(self.categorical_levels[dim])
                    cov *= coreg_cov(dim, D_out, idx)

                    # Coregion kernel for parameters, if necessary
                    if self.out_col in self.categorical_dims:
                        cov *= cov_param

                    # Combine GPs
                    gp_dict[dim] = pm_gp(cov_func=cov)
                    gp_dict["total"] += gp_dict[dim]

        self.gp_dict = gp_dict
        return gp_dict

    def build_latent(
        self,
        seed=None,
        continuous_kernel="ExpQuad",
        prior_name="latent_prior",
        ARD=True,
        lower=None,
        upper=None,
        mass=0.98,
    ):
        if self.additive:
            raise NotImplementedError("Additive/latent GPs are not yet implemented")

        X, y = self.get_shaped_data("mean")
        D_in = len(self.dims)
        assert X.shape[1] == D_in

        seed = self.seed if seed is None else seed
        self.seed = seed
        self.continuous_kernel = continuous_kernel
        self.sparse = False
        self.latent = True

        gp_dict = self._construct_kernels(
            X,
            continuous_kernel,
            seed,
            sparse=False,
            latent=True,
            ARD=ARD,
            lower=lower,
            upper=upper,
            mass=mass,
        )

        with self.model:
            self.prior = gp_dict["total"].prior(prior_name, X=X)

        return self

    def find_MAP(self, *args, **kwargs):
        """Finds maximum a posteriori value for hyperparameters in model.

        Parameters
        ----------
        *args
            Positional arguments passed to :func:`pm.find_MAP`
        **kwargs
            Keyword arguments passed to :func:`pm.find_MAP`
        """
        assert self.model is not None
        with self.model:
            self.MAP = pm.find_MAP(*args, **kwargs)

        return self.MAP

    def sample(self, *args, **kwargs):
        """Draws samples from the posterior for the hyperparameters in model.

        Parameters
        ----------
        *args
            Positional arguments passed to :func:`pm.sample`
        **kwargs
            Keyword arguments passed to :func:`pm.sample`
        """

        defaults = {
            "return_inferencedata": True,
            "random_seed": self.seed,
        }

        assert self.model is not None
        with self.model:
            self.trace = pm.sample(*args, **{**defaults, **kwargs})

        return self.trace

    def predict(self, points_array, with_noise=True, additive_level="total", **kwargs):

        # TODO: need to supply "given" dict for additive GP sublevel predictions
        if additive_level != "total":
            raise NotImplementedError("Prediction for additive sublevels is not yet supported.")

        with self.model:
            # Prediction means and variance as a numpy vector
            predictions = self.gp_dict[additive_level].predict(
                points_array, point=self.MAP, diag=True, pred_noise=with_noise, **kwargs
            )

        return predictions

    def _recursively_append(self, var_name, suffix="_", increment_var=True):
        if var_name in self.model.named_vars:
            if increment_var:
                var_name += suffix
                return self._recursively_append(var_name)
            else:
                raise ValueError(f'The variable name "{var_name}" already exists in model.')
        else:
            return var_name

    def draw_point_samples(
        self,
        points,
        *args,
        source=None,
        output=None,
        var_name="posterior_samples",
        additive_level="total",
        increment_var=True,
        **kwargs,
    ):
        """Draw posterior samples at supplied points

        Parameters
        ----------
        points : ParameterArray
            1-D ParameterArray vector of coordinates for prediction, must have one layer per ``self.dims``
        output : str or list of str, optional
            Variable for which to make predictions
        source : {None, dict, az.data.inference_data.InferenceData}
            GP parameters for which to draw samples. Should be the result of :meth:`find_MAP`, :meth:`sample`, or
            _None_.
        var_name : str, default "posterior_samples"
            Name to assign new variable to contain conditional predictions.
        additive_level : str, default "total"
            Level of additive GP at which to make predictions.
        increment_var : bool, default True
            Whether to append '_' to the end of _var_name_ if it already exists in model.

        Returns
        -------
        samples : parray
            Samples as a 'Parray'
        """

        output = self._parse_prediction_output(output)
        if len(output) > 1:
            raise NotImplementedError("Drawing correlated samples of multiple outputs is not yet implemented.")
        points_array, tall_points, param_coords = self._prepare_points_for_prediction(points, output=output)

        if source is None:
            if self.trace is None and self.MAP is None:
                raise ValueError('"Source" of predictions must be supplied if GP object has no trace or MAP stored.')
            elif self.trace is not None and self.MAP is not None:
                raise ValueError('"Source" of predictions must be supplied if GP object has both trace and MAP stored.')
            elif self.MAP is not None:
                source = [self.MAP]
            elif self.trace is not None:
                source = self.trace

        var_name = self._recursively_append(var_name, increment_var=increment_var)

        with self.model:
            _ = self.gp_dict[additive_level].conditional(var_name, points_array)

        with self.model:
            samples = pm.sample_posterior_predictive(*args, source, var_names=[var_name], **kwargs)

        self.predictions = self.parray(**{output: samples[var_name]}, stdzd=True)
        self.predictions_X = points

        return self.predictions

    def draw_grid_samples(
        self,
        *args,
        source=None,
        output=None,
        categorical_levels=None,
        var_name="posterior_samples",
        additive_level="total",
        increment_var=True,
        **kwargs,
    ):
        """Draw posterior samples at points defined by :meth:`prepare_grid`.

        Parameters
        ----------
        source : {None, dict, az.data.inference_data.InferenceData}
            GP parameters for which to draw samples. Should be the result of :meth:`find_MAP`, :meth:`sample`, or
            _None_.
        output : str or list of str, optional
            Variable(s) for which to make predictions
        categorical_levels : dict, optional
            Level for each :attr:`categorical_dims` at which to make prediction
        var_name : str, default "posterior_samples"
            Name to assign new variable to contain conditional predictions.
        additive_level : str, default "total"
            Level of additive GP at which to make predictions.
        increment_var : bool, default True
            Whether to append '_' to the end of _var_name_ if it already exists in model.

        Returns
        -------
        samples : ParameterArray
            Samples as a 'Parray' reshaped into a grid with _len(:attr:`continuous_dims`)_ dimensions
        """

        if self.grid_points is None:
            raise ValueError("Grid must first be specified with `prepare_grid`")

        points = self.grid_points
        if self.categorical_dims:
            points = self.append_categorical_points(points, categorical_levels=categorical_levels)

        samples = self.draw_point_samples(
            *args,
            points=points,
            output=output,
            source=source,
            var_name=var_name,
            additive_level=additive_level,
            increment_var=increment_var,
            **kwargs,
        )
        self.predictions = samples.reshape(-1, *self.grid_parray.shape)
        self.predictions_X = self.predictions_X.reshape(self.grid_parray.shape)

        return self.predictions
