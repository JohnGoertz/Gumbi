"""Tests behavior of structured arrays"""

import pytest
import numpy as np
import pathlib as pl

from gumbi import Standardizer
from gumbi import parray, uarray, uparray, mvuparray

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

def test_parray():
    stdzr = Standardizer(**example_stdzr, log_vars=log_vars, logit_vars=logit_vars)

    # Parameter found in stdzr
    # TODO: Update parray test when change stdzr defaults
    rpa = parray(d=np.arange(5, 10) / 10, stdzr=stdzr)
    assert np.allclose(rpa, np.arange(5, 10) / 10)
    assert np.allclose(rpa.values(), np.arange(5, 10) / 10)
    assert np.allclose(rpa.t, np.array([-0.69314718, -0.51082562, -0.35667494, -0.22314355, -0.10536052]))
    assert np.allclose(rpa.z, np.array([-2.4439695 , -1.29003559, -0.31439838,  0.53073702,  1.27619927]))

    # Composition with numpy functions
    assert np.allclose(np.min(np.sqrt(np.mean(np.square(rpa-rpa[0]-0.05)))).t, -1.5791256)
    assert np.argmax(rpa.values()) == 4

    # Parameter not found in stdzr
    pa1 = parray(param=np.arange(5), stdzr=stdzr)
    assert np.allclose(pa1, np.arange(5))
    assert np.allclose(pa1.t, np.arange(5))
    assert np.allclose(pa1.z, np.arange(5))

    # Multiple parameters
    pa2 = parray(param=np.arange(5), other=np.arange(5)*10, stdzr=stdzr)
    assert np.allclose(pa2.get('param').values(), np.array([0., 1., 2., 3., 4.]))
    assert np.allclose(pa2.get('other').values(), np.array([0., 10., 20., 30., 40.]))
    assert pa2.values().shape == (2, 5)

    # Indexing and slicing
    assert pa1[0].values() == 0
    assert np.allclose(pa1[::2].values(), np.array([0, 2, 4]))
    assert np.allclose(pa2[::2].get('param').values(), np.array([0, 2, 4]))


def test_uarray():
    ua1 = uarray('A', μ=1, σ2=0.1)
    ua2 = uarray('A', μ=2, σ2=0.2)

    # Arithmetic operations
    ua3 = ua1+1
    assert np.isclose(ua3.μ, 2.)
    assert np.isclose(ua3.σ2, 0.1)
    assert np.isclose(ua3.σ, 0.3162277660)

    ua4 = ua2+ua1
    assert np.isclose(ua4.μ, 3.)
    assert np.isclose(ua4.σ2, 0.3)

    ua5 = ua2-ua1
    assert np.isclose(ua5.μ, 1.)
    assert np.isclose(ua5.σ2, 0.3)

    ua6 = uarray.stack([ua1, ua2]).mean(axis=0)
    assert np.isclose(ua6.μ, 1.5)
    assert np.isclose(ua6.σ2, 0.075)

    ua7 = uarray('B', np.arange(1,5)/10, np.arange(1,5)/100)
    assert np.isclose(ua7.mean().μ, 0.25)
    assert np.isclose(ua7.mean().σ2, 0.00625)

    ua8 = ua1+ua7.mean()
    assert ua8.name == '(A+B)'
    assert np.isclose(ua8.μ, 1.25)
    assert np.isclose(ua8.σ2, 0.10625)

    # Conversion to scipy distribution
    assert np.allclose(ua7.dist.ppf(0.95), np.array([0.26448536, 0.43261743, 0.58489701, 0.72897073]))
    rvs = ua7.dist.rvs([3, *ua7.shape], random_state=2021)
    targets = np.array([[0.24886091, 0.29560237, 0.2275221, 0.23869584],
                        [0.15558758, 0.10022663, 0.4958704, 0.52900037],
                        [0.11064137, 0.25970171, 0.32151326, 0.23240931]])
    assert np.allclose(rvs, targets)


def test_uparray():
    stdzr = Standardizer(**example_stdzr, log_vars=log_vars, logit_vars=logit_vars)

    # TODO: Update uparray test when change stdzr defaults
    upa = uparray('c', np.arange(1, 5) / 10, np.arange(1, 5) / 100, stdzr)
    assert np.allclose(upa.μ, np.arange(1, 5) / 10)
    assert np.allclose(upa.σ2, np.arange(1, 5) / 100)

    # Round-trip transformation
    roundtrip_μ, roundtrip_σ2 = upa.stdzr.unstdz(upa.name, upa.z.μ, upa.z.σ2)
    assert np.allclose(upa.μ, roundtrip_μ)
    assert np.allclose(upa.σ2, roundtrip_σ2)

    upa2 = uparray(upa.name, upa.z.μ, upa.z.σ2, stdzr, stdzd=True)
    assert np.allclose(upa.μ, upa2.μ)
    assert np.allclose(upa.σ2, upa2.σ2)
    assert np.allclose(upa.z.μ, upa2.z.μ)
    assert np.allclose(upa.z.σ2, upa2.z.σ2)

    # Correct error propagation
    assert np.isclose(upa.mean().μ, 0.22133638)
    assert np.isclose(upa.mean().σ2, 0.00625)
    analytical_mean = upa.mean().dist.mean()
    mc_mean = np.exp(upa.t.mean().dist.rvs(10000, random_state=2021).mean())
    assert np.isclose(analytical_mean, mc_mean, atol=0.01)
    mc_var = np.log(upa.mean().dist.rvs(10000, random_state=2021)).var()
    assert np.isclose(upa.mean().σ2, mc_var, atol=1e-4)

    # Distribution behavior
    assert np.allclose(upa.dist.ppf(0.025), np.array([0.08220152, 0.1515835 , 0.21364308, 0.27028359]))
    assert np.allclose(upa.dist.ppf(0.975), np.array([0.12165225, 0.26388097, 0.42126336, 0.59197082]))
    rvs = upa.dist.rvs([3, *upa.shape], random_state=2021)
    target = np.array([[0.11605116, 0.22006429, 0.27902589, 0.34041327],
                       [0.10571616, 0.1810085 , 0.36491077, 0.45507622],
                       [0.10106982, 0.21230397, 0.3065239 , 0.33827997]])
    assert np.allclose(rvs, target)


def test_mvuparray():
    stdzr = Standardizer(**example_stdzr, log_vars=log_vars, logit_vars=logit_vars)

    # TODO: Update uparray test when change stdzr defaults
    stdzr = Standardizer(**example_stdzr, log_vars=log_vars, logit_vars=logit_vars)
    c_μ = np.arange(1, 5)/10
    c_σ2 = np.arange(1, 5)/100
    r_μ = c_μ+0.5
    r_σ2 = c_σ2/100*2
    c_upa = uparray('c', c_μ, c_σ2, stdzr)
    r_upa = uparray('d', r_μ, r_σ2, stdzr)
    cor = np.array([[1, -0.6], [-0.6, 1]])
    mvup = mvuparray(c_upa, r_upa, cor=cor)

    # Proper construction
    assert np.allclose(mvup.μ.values(), np.stack([c_μ, r_μ]))
    assert np.allclose(mvup.get('d').μ, r_μ)
    assert np.allclose(mvup.t.get('d_t').μ, r_upa.t.μ)
    assert np.allclose(mvup.t.μ['d_t'], r_upa.t.μ)
    assert np.allclose(mvup.z.μ['d_z'], r_upa.z.μ)

    # Distribution behavior
    pa = mvup.parray(c=0.09, d=0.61)
    assert np.isclose(mvup[0].dist.cdf(pa), 0.0889634198684274)
    assert np.isclose(mvup[0].t.dist.cdf(pa), 0.0889634198684274)
    assert np.isclose(mvup[0].z.dist.cdf(pa), 0.0889634198684274)
    mvup[0].dist.cdf(mvup.μ)  # Should run without errors

    rvs = mvup[0].dist.rvs(2, random_state=2021)
    assert np.allclose(rvs['d'].values(), np.array([0.61310678, 0.59268474]))
    assert np.allclose(rvs['c'].values(), np.array([0.08709257, 0.10308707]))
