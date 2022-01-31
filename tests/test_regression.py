import pytest
import pickle
import pandas as pd
import numpy as np
import pathlib as pl
import pymc3 as pm

from gumbi import GP, DataSet, Standardizer

test_dir = pl.Path(__file__).resolve().parent
test_data = test_dir / 'test_data'

example_stdzr = {
    'a': {'μ': -0.762, 'σ2': 1.258**2},
    'b': {'μ': -0.0368, 'σ2': 0.351**2},
    'c': {'μ': -5.30, 'σ2': 0.582**2},
    'd': {'μ': -0.307, 'σ2': 0.158**2},
    'e': {'μ': -1.056, 'σ2': 0.398**2},
    'f': {'μ': 3.34, 'σ2': 0.1501**2},
    'X': {'μ': -0.282, 'σ2': 1**2},
    'Y': {'μ': 4.48, 'σ2': 0.75**2},
    'lg10_Z': {'μ': 5, 'σ2': 2**2},
}

log_vars = ['d', 'f', 'b', 'c', 'Y']
logit_vars = ['e', 'X']

################################################################################
# GP
################################################################################

@pytest.fixture
def example_estimates():
    es = pd.read_pickle(test_data / 'test_dataset.pkl')
    stdzr = Standardizer(**example_stdzr, log_vars=log_vars, logit_vars=logit_vars)
    ds = DataSet.from_tidy(es, names_column='Parameter', stdzr=stdzr)
    return ds

@pytest.fixture
def example_gp(example_estimates):
    return GP(example_estimates, outputs='d')

# Input argument parsing

def test_gp_default_fit_parsing(example_gp):
    gp = example_gp.specify_model(continuous_dims=['X', 'Y'])
    assert gp.continuous_dims == ['X', 'Y']
    assert gp.categorical_dims == []


def test_gp_data_parsing(example_gp):
    gp = example_gp.specify_model(continuous_dims=['X', 'Y'])
    X, y = gp.get_structured_data()
    assert X.shape == (66,)
    assert len(X.names) == 2
    assert y.shape == (66,)


def test_gp_numerical_continuous_fit_parsing(example_gp):
    gp = example_gp.specify_model(continuous_dims=['X', 'Y', 'lg10_Z'])
    assert gp.continuous_dims == ['X', 'Y', 'lg10_Z']
    for dim in gp.continuous_dims:
        assert len(gp.continuous_levels[dim]) == len(gp.data.tidy[dim].unique())
        assert len(gp.continuous_coords[dim].values()) == len(gp.continuous_levels[dim])
    assert gp.categorical_dims == []
    X, y = gp.get_structured_data()
    assert X.shape == (66,)
    assert len(X.names) == 3
    assert y.shape == (66,)


def test_gp_categorical_continuous_fit_parsing(example_gp):
    gp = example_gp.specify_model(continuous_dims=['X', 'Y', 'Name'])
    assert gp.continuous_dims == ['X', 'Y', 'Name']
    for dim in gp.continuous_dims:
        assert len(gp.continuous_levels[dim]) == len(gp.data.tidy[dim].unique())
        assert len(gp.continuous_coords[dim].values()) == len(gp.continuous_levels[dim])
    assert gp.categorical_dims == []
    X, y = gp.get_structured_data()
    assert X.shape == (66,)
    assert len(X.names) == 3
    assert y.shape == (66,)


def test_gp_params_fit_parsing(example_gp):
    gp = example_gp.specify_model(outputs=['d', 'c'], continuous_dims=['X', 'Y'])
    assert gp.continuous_dims == ['X', 'Y']
    assert gp.categorical_dims == ['Parameter']
    assert gp.categorical_levels == {'Parameter': ['d', 'c']}
    assert gp.categorical_coords == {'Parameter': {'d': 1, 'c': 0}}
    X, y = gp.get_structured_data()
    assert X.shape == (66,)
    assert len(X.names) == 2
    assert y.shape == (66,)
    assert len(y.names) == 2


def test_gp_single_input_fit_parsing(example_gp):
    gp = example_gp.specify_model(continuous_dims=['X', 'Y', 'Name'], continuous_levels={'Name': ['intense-opportunity']})
    assert gp.continuous_dims == ['X', 'Y']
    assert gp.filter_dims == {'Name': ['intense-opportunity'], 'Parameter': ['d']}
    X, y = gp.get_structured_data()
    assert X.shape == (7,)
    assert len(X.names) == 2
    assert y.shape == (7,)

# Model building

def test_gp_build_model_simple(example_gp):
    gp = example_gp.specify_model(continuous_dims=['X', 'Y'])
    gp.build_model()
    assert isinstance(gp.model, pm.model.Model)
    assert isinstance(gp.gp_dict['total'], pm.gp.gp.Marginal)


# Combinatorial gp objects with various parameterizations
@pytest.fixture(params=[False, True])
def additive(request):
    return request.param


@pytest.fixture(params=[{'outputs': ['d', 'c'], 'continuous_dims': ['X', 'Y']},
                        {'continuous_dims': ['X', 'Y'], 'categorical_dims': 'Code'},
                        {'continuous_dims': ['X', 'Y', 'Name']},
                        {'continuous_dims': ['X', 'Y', 'lg10_Z']},
                        {'continuous_dims': ['X', 'Y', 'Name'], 'continuous_levels': {'Name': ['intense-opportunity']}}])
def fit_inputs(request):
    return request.param


@pytest.mark.slow
def test_gp_build_model(example_gp, fit_inputs, additive):
    # Basically just makes sure that `build_model` runs without errors
    gp = example_gp.specify_model(**fit_inputs, additive=additive)
    gp.build_model()
    assert isinstance(gp.model, pm.model.Model)
    assert isinstance(gp.gp_dict['total'], pm.gp.gp.Marginal)


@pytest.mark.slow
def test_gp_build_model_additive(example_gp):
    # Basically just makes sure that `build_model` runs without errors
    gp = example_gp.specify_model(outputs=['d', 'c'], continuous_dims=['X', 'Y'], categorical_dims='lg10_Z', additive=True)
    gp.build_model()
    assert isinstance(gp.model, pm.model.Model)
    assert all(name in gp.gp_dict.keys() for name in ['total', 'global', 'lg10_Z'])

# MAP estimation

def test_gp_fit_simple(example_gp):
    # Basically just makes sure that `fit` runs without errors
    gp = example_gp.fit(continuous_dims=['X', 'Y', 'lg10_Z'], continuous_levels={'lg10_Z': [8]})
    assert isinstance(gp.MAP, dict)


@pytest.mark.slow
def test_gp_fit(example_gp, fit_inputs, additive):
    # Basically just makes sure that `fit` runs without errors
    gp = example_gp.fit(**fit_inputs, additive=additive)
    assert isinstance(gp.MAP, dict)
