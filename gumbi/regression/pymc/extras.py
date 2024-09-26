from functools import wraps

import pymc as pm

from .GP import PymcGP


class PymcGPC(PymcGP):
    @wraps(PymcGP.build_model)
    def build_model(
        self,
        seed=None,
        continuous_kernel="ExpQuad",
        heteroskedastic_inputs=False,
        heteroskedastic_outputs=False,
        sparse=False,
        n_u=100,
        eps=1e-6,
    ):
        if heteroskedastic_inputs:
            raise NotImplementedError("The PymcGP Classifier does not support heteroskedastic inputs.")
        if heteroskedastic_outputs:
            raise NotImplementedError("The PymcGP Classifier does not support heteroskedastic outputs.")
        if sparse:
            raise NotImplementedError("The PymcGP Classifier does not support sparse structure (yet).")

        self.build_latent(seed=seed, continuous_kernel=continuous_kernel, eps=eps)

        _, y = self.get_shaped_data("mean")

        with self.model:
            f = self.prior

            # logit link and Bernoulli likelihood
            p = pm.Deterministic("p", pm.math.invlogit(f))
            _ = pm.Bernoulli("y", p=p, observed=y)

        return self

    @wraps(PymcGP.draw_point_samples)
    def draw_point_samples(
        self,
        points,
        *args,
        source=None,
        output=None,
        var_name="posterior_samples",
        additive_level="total",
        increment_var=True,
        **kwargs
    ):
        var_name = self._recursively_append(var_name, increment_var=increment_var)

        # A
        self.stdzr.logit_vars += [var_name]

        return super(PymcGPC, self).draw_point_samples(
            points,
            *args,
            source=source,
            output=output,
            var_name=var_name,
            additive_level=additive_level,
            increment_var=True,
            **kwargs
        )
