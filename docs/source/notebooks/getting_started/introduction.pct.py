# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,.pct.py:percent
#     notebook_metadata_filter: kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
# ---

# %% [markdown]
# # Introduction
#
# Gumbi simplifies the steps needed to build a Gaussian Process model from tabular data. It takes care of shaping,
# transforming, and standardizing data as necessary while applying best practices and sensible defaults to the
# construction of the GP model itself. Taking inspiration from popular packages such as
# __[Bambi](https://bambinos.github.io/bambi/main/index.html)__ and
# __[Seaborn](https://seaborn.pydata.org/index.html)__, Gumbi's aim is to allow quick iteration on both model structure
# and prediction visualization. Gumbi is primarily built on top of __[Pymc](https://docs.pymc.io/)__, though additional
# support for __[GPflow](https://gpflow.readthedocs.io/)__ is planned.

# %% [markdown]
# ## Quickstart

# %% [markdown] Read in some data and store it as a Gumbi `DataSet`:
#

import seaborn as sns

# %%
import gumbi as gmb

cars = sns.load_dataset("mpg").dropna()
ds = gmb.DataSet(
    cars,
    outputs=["mpg", "acceleration"],
    log_vars=["mpg", "acceleration", "horsepower"],
)

# %% [markdown]
#
# Create a Gumbi `GP` object and fit a model that predicts *mpg* from *horsepower*:

# %%
gp = gmb.GP(ds)
gp.fit(outputs=["mpg"], continuous_dims=["horsepower"])

# %% [markdown]
#
# Make predictions and plot!

# %%
X = gp.prepare_grid()
y = gp.predict_grid()
gmb.ParrayPlotter(X, y).plot()
sns.scatterplot(data=cars, x="horsepower", y="mpg", color=sns.cubehelix_palette()[-1], alpha=0.5)

# %% [markdown]
#
# More complex GPs are also possible, such as correlated multi-input and multi-output systems, demonstrated in the
# example notebooks.

# %% [markdown]
# ## Installation
# ### Via pip
#
#     pip install gumbi
#
# ### Bleeding edge
#
#     pip install git+git://github.com/JohnGoertz/Gumbi.git@develop
#
# ### Local development
#
# * Clone the repo and navigate to the new directory
#
#   * `git clone https://gitlab.com/JohnGoertz/gumbi gumbi`
#
#   * `cd gumbi`
#
# * Create a new conda environment using mamba
#
#   * `conda install mamba`
#
#   * `mamba install -f gumbi_env.yaml`
#
# * Install `gumbi` via `pip` in editable/development mode
#
#   * From within the `gumbi` repo
#
#   * `pip install --editable ./`
#
# * To update the `gumbi` module
#
#   * From within the `gumbi` repo
#
#   * `git pull`
#
