from gumbi import Regressor, ParameterArray
from gumbi.utils import listify, assert_in
from gumbi.utils.gp_utils import get_ls_prior, GPyTorchInverseGammaPrior
import numpy as np

import torch
from torch.nn.functional import pdist
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, MixedSingleTaskGP, HeteroskedasticSingleTaskGP, MultiTaskGP, KroneckerMultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize as bt_unnormalize
from botorch.utils.transforms import normalize as bt_normalize
from gpytorch.kernels import RBFKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch

from warnings import simplefilter
from linear_operator.utils.warnings import NumericalWarning

import warnings


class BotorchGP(Regressor):

    def __init__(self, *args, gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and gpu else "cpu"
        )
        self.model = None
        self.mll = None
        self.structure = None
        self.optimization_bounds = None

    def fit(
        self,
        outputs=None,
        *,
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
        heteroskedastic_outputs=False,
        sparse=False,
        n_u=100,
        ARD=False,
        ls_bounds=None,
        spec_kwargs=None,
        build_kwargs=None,
        multitask_kernel=None,
        **kwargs,
    ):

        if additive:
            raise NotImplementedError(
                "Additive models are not yet supported in BotorchGP."
            )
        if period is not None:
            raise NotImplementedError(
                "Periodic kernels are not yet supported in BotorchGP."
            )
        if linear_dims is not None:
            warnings.warn(
                "Linear dimensions are not yet supported in BotorchGP and will be ignored."
            )
        if categorical_dims is not None and len(listify(outputs)) > 1:
            raise NotImplementedError(
                "Categorical dimensions with multiple outputs are not yet supported in BotorchGP."
            )

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
            multitask_kernel=multitask_kernel,
            **(build_kwargs or {}),
        )

        return self.fit_model(**kwargs)

    def build_model(
        self,
        *,
        continuous_kernel="ExpQuad",
        period=None,
        heteroskedastic_inputs=False,
        heteroskedastic_outputs=False,
        sparse=False,
        n_u=100,
        ARD=False,
        ls_bounds=None,
        seed=None,
        mass=0.98,
        multitask_kernel=None
    ):

        if period is not None:
            raise NotImplementedError(
                "Periodic kernels are not yet supported in BotorchGP."
            )
        if sparse:
            raise NotImplementedError(
                "Sparse models are not yet supported in BotorchGP."
            )
        if heteroskedastic_inputs:
            raise NotImplementedError(
                "Heteroskedasticity over inputs is not yet implemented in BotorchGP."
            )
        if heteroskedastic_outputs:
            raise NotImplementedError(
                "Heteroskedasticity over inputs is not yet implemented in BotorchGP."
            )
        # if ls_bounds is not None:
        #     raise NotImplementedError(
        #         "User-defined lengthscale bounds are not yet implemented in BotorchGP."
        #     )
        # if mass != self.build_model.__kwdefaults__["mass"]:
        #     warnings.warn(f"`mass` keyword ignored in {self.__class__.__name__}")
            
        if multitask_kernel is not None:
            multitask_kernel = multitask_kernel.capitalize()
        assert_in('multitask_kernel', multitask_kernel, [None, 'Kronecker','Hadamard'])

        X, y = self.get_shaped_data("mean")

        D_in = len([d for d in self.dims if d != self.out_col])
        D_out = len(self.outputs)
        if D_out == 1:
            assert X.shape[1] == D_in
        else:
            assert X.shape[1] == D_in + 1

        seed = self.seed if seed is None else seed
        self.seed = seed
        self.continuous_kernel = continuous_kernel
        self.heteroskedastic_inputs = heteroskedastic_inputs
        self.heteroskedastic_outputs = heteroskedastic_outputs
        self.sparse = sparse
        self.n_u = n_u
        self.latent = False

        Yvar = None
            
        if D_out == 1:
            cat_dims = [d for d in self.categorical_dims if d != self.out_col]
            n_cats = len(cat_dims)
            if n_cats == 0:
                if heteroskedastic_inputs:
                    self.structure = 'HeteroskedasticSingleTask'
                    self.model = HeteroskedasticSingleTaskGP(
                        X,
                        y,
                        train_Yvar=Yvar,
                        # covar_module=self._get_kernel(
                        #     continuous_kernel, ARD, mass=mass, ls_bounds=ls_bounds
                        # ),
                        input_transform=Normalize(d=D_in),
                    ).to(self.device)
                else:
                    self.structure = 'SingleTask'
                    self.model = SingleTaskGP(
                        X,
                        y,
                        train_Yvar=Yvar,
                        covar_module=self._get_kernel(
                            continuous_kernel, ARD, mass=mass, ls_bounds=ls_bounds
                        ),
                        input_transform=Normalize(d=D_in),
                    ).to(self.device)
            else:
                self.structure = 'MixedSingleTaskGP'
                cat_idxs = [self.dims.index(dim) for dim in cat_dims]
                self.model = MixedSingleTaskGP(
                    X,
                    y,
                    cat_dims=cat_idxs,
                    train_Yvar=Yvar,
                    cont_kernel_factory=self._get_kernel_factory(
                        continuous_kernel, ARD, mass=mass, ls_bounds=ls_bounds
                    ),
                    input_transform=Normalize(d=D_in),
                ).to(self.device)
        else:
            # Decide between Kronecker and Hadamard structure
            Xs, ys = self.get_separated_data("mean")
            kronecker_possible = all([Xs[0].shape == x.shape for x in Xs[1:]])
            if kronecker_possible:
                kronecker_possible = all([torch.allclose(Xs[0], x) for x in Xs[1:]])
            kronecker_default = D_out > 2
            force_kronecker = multitask_kernel == 'Kronecker'
            if kronecker_possible and (kronecker_default or force_kronecker):
                # Kronecker structure: all outputs observed at every input point
                # Allows performance improvements for high-dimensional outputs
                self.structure = 'KroneckerMultiTaskGP'
                self.model = KroneckerMultiTaskGP(
                    Xs[0],
                    torch.cat(ys, dim=1),
                    data_covar_module=self._get_kernel(
                        continuous_kernel, ARD, mass=mass, ls_bounds=ls_bounds
                    ),
                    input_transform=Normalize(d=D_in),
                ).to(self.device)
            else:
                # Hadamard structure: outputs observed at different input points
                self.structure = 'MultiTaskGP'
                self.model = MultiTaskGP(
                    X,
                    y,
                    task_feature=-1,
                    train_Yvar=Yvar,
                    covar_module=self._get_kernel(
                        continuous_kernel, ARD, mass=mass, ls_bounds=ls_bounds
                    ),
                    input_transform=Normalize(d=D_in + 1),
                ).to(self.device)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        return self.model, self.mll

    def get_shaped_data(self, metric="mean", dropna=True):
        X, y = super().get_shaped_data(metric=metric, dropna=dropna)

        X = torch.tensor(X).double().to(self.device)
        y = torch.tensor(y).double().to(self.device)
        return X, y.unsqueeze(-1)

    def get_separated_data(self, metric="mean", dropna=True):
        X, y = self.get_shaped_data(metric=metric, dropna=dropna)
        if len(self.outputs) == 1:
            return [X], [y]
        else:
            idxs = torch.unique(X[:, -1])
            Xs, ys = [], []
            for i in idxs:
                idx = X[:, -1] == i
                Xs.append(X[idx, :-1])
                ys.append(y[idx])
            return Xs, ys
        
    def _get_kernel_factory(self, kernel_type, ard, mass=0.98, ls_bounds=None):

        # Apply lengthscale constraints if provided
        default_lower, default_upper = self._get_default_bounds(ard)
        if ls_bounds is not None:
            lower, upper = ls_bounds
            lower = np.maximum(lower, default_lower)
            upper = np.minimum(upper, default_upper)
        else:
            lower, upper = default_lower, default_upper

        lengthscale_constraint = gpytorch.constraints.Interval(lower, upper)

        Xs, _ = self.get_separated_data("mean")
        X = torch.cat(Xs, dim=0)
        lengthscale_prior_params = get_ls_prior(X, ARD=ard, lower=lower, upper=upper, mass=mass, dist='InverseGamma')
        alpha = torch.tensor(lengthscale_prior_params['alpha'])
        beta = torch.tensor(lengthscale_prior_params['beta'])
        lengthscale_prior = GPyTorchInverseGammaPrior(concentration=alpha, rate=beta)
        
        def kernel_factory(*, batch_shape=None, ard_num_dims=None, active_dims=None):

            match kernel_type:
                case "RBF" | "ExpQuad":
                    base_kernel = RBFKernel(
                        lengthscale_constraint=lengthscale_constraint,
                        lengthscale_prior=lengthscale_prior,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        active_dims=active_dims,
                    )
                case "Matern32" | "Matern3/2":
                    base_kernel = MaternKernel(
                        nu=1.5,
                        lengthscale_constraint=lengthscale_constraint,
                        lengthscale_prior=lengthscale_prior,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        active_dims=active_dims,
                    )
                case "Matern52" | "Matern5/2":
                    base_kernel = MaternKernel(
                        nu=2.5,
                        lengthscale_constraint=lengthscale_constraint,
                        lengthscale_prior=lengthscale_prior,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        active_dims=active_dims,
                    )
                case _:
                    raise ValueError(
                        "Invalid kernel type. Choose 'RBF', 'ExpQuad', 'Matern32', 'Matern3/2', 'Matern5/2', or 'Matern52'."
                    )
            return base_kernel
        return kernel_factory

    def _get_kernel(self, continuous_kernel, ard, mass=0.98, ls_bounds=None):
        """
        Returns the appropriate kernel based on user settings.
        """
                
        kernel_factory = self._get_kernel_factory(continuous_kernel, ard, mass=mass, ls_bounds=ls_bounds)
        D_in = len([d for d in self.dims if d != self.out_col])
        base_kernel = kernel_factory(ard_num_dims=D_in if ard else None)

        return base_kernel

    def _get_default_bounds(self, ard: bool):
        def _get_default_lower_upper(x):

            distances = pdist(x)
            distinct = distances != 0

            default_lower = distances[distinct].min() if sum(distinct) > 0 else 0.01
            default_upper = distances[distinct].max() if sum(distinct) > 0 else 1

            return default_lower, default_upper

        Xs, _ = self.get_separated_data("mean")
        X = torch.cat(Xs, dim=0)

        if ard:
            lengthscale_bounds = torch.stack(
                [torch.tensor(_get_default_lower_upper(X_.unsqueeze(1))) for X_ in X.T]
            ).T
            lower, upper = lengthscale_bounds.cpu().numpy().squeeze().tolist()
        else:
            lengthscale_bounds = torch.tensor(_get_default_lower_upper(X))
            lower, upper = lengthscale_bounds.cpu().numpy().squeeze().tolist()

        return lower, upper

    def fit_model(self, **kwargs):
        """
        Fits the GP model using exact marginal log likelihood.
        """
        torch.manual_seed(self.seed)
        return fit_gpytorch_mll(self.mll, **kwargs)

    def predict(self, points_array, *, with_noise=True, additive_level="total"):
        """
        Make predictions at supplied points using specified gp

        Parameters
        ----------
        points_array : np.ndarray
            Array of points at which to make predictions
        with_noise : bool, optional
            Whether to include noise in the predictions, by default True

        Returns
        -------
        mean, variance : tuple of np.ndarray
            Mean and variance of predictions at each supplied points
        """
        points_tensor = torch.tensor(points_array).float().to(self.device)

        if additive_level != "total":
            raise NotImplementedError(
                "Prediction for additive sublevels is not yet supported."
            )

        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            t_points_tensor = self.model.input_transform.transform(points_tensor)
            predictive_distribution = self.model.likelihood(self.model(t_points_tensor))
            self.predictive_distribution = predictive_distribution
            if with_noise:
                # The likelihood will add noise to the posterior
                mean = predictive_distribution.mean
                variance = predictive_distribution.stddev**2
                # lower, upper = predictive_distribution.confidence_region()
            else:
                posterior = self.model.posterior(points_tensor)
                mean = posterior.mean
                variance = posterior.variance

        return mean.cpu().numpy(), variance.cpu().numpy()

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

        # TODO: add convenience method for predicting at a single point

        output = self._parse_prediction_output(output)
        D_out = len(output)
        points_array, tall_points, param_coords = self._prepare_points_for_prediction(
            points, output=output
        )
        
        if self.structure == 'KroneckerMultiTaskGP':
            # Kronecker structure: all outputs observed at every input point
            points_array = points_array[points_array[:,-1]==0, :-1]

        # Prediction means and variance as a list of numpy vectors
        pred_mean, pred_variance = self.predict(
            points_array, with_noise=with_noise, **kwargs
        )
        self.predictions_X = points
        
        if self.structure == 'KroneckerMultiTaskGP':
            pred_mean

        # Store predictions in appropriate structured array format
        if D_out == 1:
            # Predicting one output, return an UncertainParameterArray
            self.predictions = self.uparray(
                output[0], pred_mean, pred_variance, stdzd=True
            )
        else:
            # Predicting multiple parameters, return an MVUncertainParameterArray First split prediction into
            # UncertainParameterArrays
            uparrays = []
            out_coords = {v:k for k, v in self.categorical_coords[self.out_col].items()}
            for i in range(D_out):
                if self.structure != 'KroneckerMultiTaskGP':
                    # idx = (tall_points[self.out_col].values() == param_coords[i]).squeeze()
                    idx = points_array[:,-1] == i
                    μ = pred_mean[idx]
                    σ2 = pred_variance[idx]
                    name = out_coords[i]
                else:
                    μ = pred_mean[:,i]
                    σ2 = pred_variance[:,i]
                    name = out_coords[i]
                uparrays.append(self.uparray(name, μ, σ2, stdzd=True))

            # Calculate the correlation matrix from the hyperparameters of the coregion kernel
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                warnings.catch_warnings(),
            ):
                simplefilter("ignore", category=NumericalWarning)
                if self.structure == 'KroneckerMultiTaskGP':
                    task_covar_module = self.model.covar_module.task_covar_module
                else:
                    task_covar_module = self.model.task_covar_module

                W = task_covar_module.covar_factor
                κ = task_covar_module.var

                cov = W @ W.T + torch.diag(κ)

                σ_task = torch.sqrt(torch.diag(cov))

                σ_outer = σ_task.unsqueeze(0) * σ_task.unsqueeze(1)  # Outer product
                cor = cov / σ_outer

            cor = cor.cpu().numpy()

            # Store predictions as MVUncertainParameterArray
            self.predictions = self.mvuparray(*uparrays, cor=cor)

        return self.predictions

    def propose(
        self,
        *,
        q=1,
        bounds=None,
        maximize=True,
        num_restarts=10,
        raw_samples=100,
        mc_samples=256,
        seed=None,
        ref_point=None,
        sequential=False,
    ):
        """
        Optimizes the acquisition function (Expected Improvement) to propose new candidates.
        """

        seed = seed if seed is not None else self.seed
        torch.manual_seed(seed)

        qmc_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([mc_samples]), seed=seed
        )

        X, y = self.get_shaped_data("mean")
        D_out = len(self.outputs)

        if bounds is None:
            if D_out == 1:
                bounds = torch.stack([X.min(0).values, X.max(0).values]).to(self.device)
            else:
                bounds = torch.stack(
                    [X[:, :-1].min(0).values, X[:, :-1].max(0).values]
                ).to(self.device)
        else:
            if isinstance(bounds, ParameterArray):
                bounds = bounds.z.values().T
            bounds = torch.tensor(bounds).to(self.device)
            
        self.optimization_bounds = bounds

        if D_out == 1:
            # Construct the objective function for the acquisition function

            def identity(samples, X=None):
                return samples.squeeze(-1)

            def neg_identity(samples, X=None):
                return -identity(samples, X)  # Negate the samples for minimization

            if maximize:
                objective = GenericMCObjective(identity)
            else:
                objective = GenericMCObjective(neg_identity)
            
            with warnings.catch_warnings():
                simplefilter("ignore", category=NumericalWarning)

                # Expected Improvement acquisition function
                acq_func = qLogNoisyExpectedImprovement(
                    model=self.model,
                    X_baseline=X,
                    sampler=qmc_sampler,
                    objective=objective,
                )
        else:

            Xs, ys = self.get_separated_data("mean")
            X_baseline = torch.unique(torch.cat(Xs, dim=0), dim=0)

            if ref_point is None:
                ref_point = []
                for y_ in ys:
                    if maximize:
                        ref_point.append(y_.min().item() - 1e-3)
                    else:
                        ref_point.append(y_.max().item() + 1e-3)

            self.ref_point = ref_point
            
            with warnings.catch_warnings():
                simplefilter("ignore", category=NumericalWarning)

                # Expected Hyper-Volume Improvement acquisition function
                acq_func = qLogNoisyExpectedHypervolumeImprovement(
                    model=self.model,
                    ref_point=torch.tensor(ref_point),  # use known reference point
                    X_baseline=bt_normalize(X_baseline, bounds),
                    prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
                    sampler=qmc_sampler,
                    cache_root=False,
                )

                bounds = bt_normalize(bounds, bounds)

        with warnings.catch_warnings():
            simplefilter("ignore", category=NumericalWarning)

            # Optimize acquisition function to propose new candidates
            candidates, acqf_values = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=q,  # Batch size
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                sequential=sequential,
            )

        if D_out > 1:
            # Unnormalize the candidates
            candidates = bt_unnormalize(candidates, bounds)

        candidates, acqf_values = candidates.cpu().numpy(), acqf_values.cpu().numpy()
        candidates = self.parray(
            **{dim: values for dim, values in zip(self.continuous_dims, candidates.T)},
            stdzd=True,
        )
        return candidates, acqf_values