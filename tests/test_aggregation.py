import pytest
import pickle
import pandas as pd
import numpy as np
import pathlib as pl
import pymc3 as pm

from gumbi import Standardizer, WideData, DataSet

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
# Standardizer
################################################################################

def test_stdz():
    s = Standardizer(**example_stdzr, log_vars=log_vars, logit_vars=logit_vars)
    nat_defaults = {p: s.untransform(p, v['μ']) for p, v in example_stdzr.items()}
    assert np.allclose([s.stdz(p, nat_defaults[p]) for p, v in nat_defaults.items()], 0), 'Standardization failed'
    assert np.allclose([s.unstdz(p, s.stdz(p, nat_defaults[p])) for p, v in example_stdzr.items()],
                       [v for v in nat_defaults.values()]), 'Round-trip standardization failed'

    stdzr = Standardizer(x={'μ': 1, 'σ2': 0.1}, d={'μ': 0, 'σ2': 0.1}, log_vars=['d'])
    assert stdzr.transform('x', μ=1) == 1
    assert stdzr.stdz('x', 1) == 0.0
    assert stdzr.unstdz('x', 0) == 1.0
    assert np.isclose(stdzr.stdz('x', 1+0.1**0.5), 1.)
    assert np.isclose(stdzr.unstdz('x', 1), 1+0.1**0.5)
    assert stdzr.stdz('d', 1) == 0.0
    assert np.isclose(stdzr.stdz('d', np.exp(0.1**0.5)), 1.0)

    assert stdzr.transform('x', μ=1, σ2=0.1) == (1, 0.1)
    assert stdzr.stdz('x', 1, 0.1) == (0.0, 1.0)
    assert stdzr.stdz('d', 1, 0.1) == (0.0, 1.0)
    assert stdzr.transform('d', 1, 0.1) == (0.0, 0.1)

    x_series = pd.Series(np.arange(1, 5), name='x')
    x_stdzd = np.array([0.0, 3.162278, 6.324555, 9.486833])
    assert np.allclose(stdzr.stdz(x_series).values, x_stdzd)

    r_series = pd.Series(np.arange(1, 5), name='d')
    r_stdzd = np.array([0.0, 2.19192384, 3.4741171 , 4.38384769])
    assert np.allclose(stdzr.stdz(r_series).values, r_stdzd)

################################################################################
# DataSet
################################################################################

@pytest.fixture
def example_dataset():
    df = pd.read_pickle(test_data / 'estimates_test_data.pkl')
    ds = DataSet.from_tidy(df, names_column='Parameter', log_vars=['Y', 'c', 'b'], logit_vars=['X', 'e'])
    assert ds.wide is not None
    assert ds.wide.z is not None
    assert ds.tidy is not None
    assert ds.tidy.z is not None
    return ds


def test_tidy_z(example_dataset):
    ds = example_dataset
    assert ds.tidy.z.shape == ds.tidy.shape
    assert np.allclose([ds.tidy.z[ds.tidy.z.Parameter == p]['Value'].mean() for p in ds.tidy.z.Parameter.unique()], 0)


def test_wide_io(example_dataset):
    ds = example_dataset

    wide_out = ds.wide
    wide_in_wd = WideData(wide_out, outputs=ds.outputs, log_vars=['Y', 'c', 'b'], logit_vars=['X', 'e'])
    wide_in_ds = DataSet(wide_out, outputs=ds.outputs, log_vars=['Y', 'c', 'b'], logit_vars=['X', 'e'])

    pd.testing.assert_frame_equal(wide_in_wd, wide_out)
    pd.testing.assert_frame_equal(wide_in_wd, wide_in_ds.wide)
    pd.testing.assert_frame_equal(ds.wide, wide_in_ds.wide)
    pd.testing.assert_frame_equal(ds.wide, wide_in_wd)

    ds.wide = wide_out.drop(0)
    pd.testing.assert_frame_equal(ds.wide, wide_out.drop(0))
    pd.testing.assert_frame_equal(ds.wide, wide_in_wd.drop(0))

