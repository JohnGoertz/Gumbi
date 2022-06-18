# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#     notebook_metadata_filter: kernelspec
# ---

# %% [markdown]
# # Multioutput Regression

import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
import seaborn as sns

import gumbi as gmb

# %% [markdown] Use gumbi's plotting defaults for stylistic consistency, good dataviz practice, and aesthetics. Use
# `gmb.style.default` if you don't have the Futura font installed.

# %%
plt.style.use(gmb.style.futura)

# %% [markdown]
# ## Setup

# %% [markdown]
#
# Load in wide-form data and store as a DataSet. We must specify which columns are considered "outputs", and
# additionally we may indicate which input and output variables should be treated as log-normal or logit-normal.

# %%
df = pd.read_pickle(gmb.data.example_dataset)
df = df[(df.Name == "binary-pollen") & (df.Color == "cyan") & (df.Metric == "mean")]
outputs = ["a", "b", "c", "d", "e", "f"]
log_vars = ["Y", "b", "c", "d", "f"]
logit_vars = ["X", "e"]
ds = gmb.DataSet(df, outputs=outputs, log_vars=log_vars, logit_vars=logit_vars)
ds

# %% [markdown]
# ## Train Model

# %% [markdown]
#
# Train a multioutput GP on five amplification parameters, using an RBF + linear kernel on `lg10_Z` to capture
# continuous variation along with a coregionalization kernel to allow for correlated but distinct behavior in both the
# mean and noise across the parameters.

# %%
fit_params = ["a", "b", "c", "d", "e"]
gp = gmb.GP(ds, outputs=fit_params)
n_p = len(fit_params)
gp.fit(continuous_dims="lg10_Z", linear_dims="lg10_Z")

# %% [markdown]
# ## Marginal Parameter Predictions

# %% [markdown] Predict each parameter individually, allowing us to ensure the marginal uncertainty is calibrated
# correctly.

# %%
limits = gp.parray(lg10_Z=[1, 9])
x_pa = gp.prepare_grid(limits=limits, resolution=17)

axs = plt.subplots(1, n_p, figsize=(n_p * 5, 4))[1]

for ax, param in zip(np.atleast_1d(axs), fit_params):
    gp.predict_grid(output=param)

    y_upa = gp.predictions

    gmb.ParrayPlotter(x_pa, y_upa).plot(ax=ax)

    param_data = ds.tidy[(ds.tidy.Metric == "mean") & (ds.tidy.Variable == param)]
    x = gp.parray(lg10_Z=param_data["lg10_Z"])
    y = param_data["Value"]
    ax.plot(x, y, "o", color=sns.cubehelix_palette()[-3])

plt.tight_layout()

# %% [markdown]
# ## Correlated Parameter Predictions

# %% [markdown] Make joint predictions for all parameters, returning an MVUncertainParameterArray.

# %%
gp.prepare_grid(limits=gp.parray(lg10_Z=[1, 9]), resolution=5)
gp.predict_grid()

x_pa = gp.predictions_X
mvup = gp.predictions
mvup

# %% [markdown] Sample from the joint distribution at each concentration to inspect the correlations between parameters
# across concentrations.

# %%
samples_df = pd.concat(
    [
        pd.DataFrame(point.dist.rvs(1000, random_state=i).as_dict()).assign(
            lg10_Z=copies.values()
        )
        for i, (point, copies) in enumerate(zip(mvup, x_pa))
    ],
    ignore_index=True,
)

sns.pairplot(samples_df, hue="lg10_Z", kind="kde", corner=True, plot_kws={"levels": 1})

# %%
