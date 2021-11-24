"""Tests behavior of structured arrays"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from gumbi import *

def test_1d_array_plotting():
    stdzr = Standardizer(c={'μ': -5.30, 'σ': 0.582}, d={'μ': -0.307, 'σ': 0.158}, z={'μ': 2, 'σ': 2},
                         log_vars=['d', 'c'], logit_vars=['z'])
    m_pa = parray(m=np.arange(1, 5) / 10, stdzr=stdzr)
    r_upa = uparray('d', np.arange(1, 5) / 10 + 0.5, np.arange(1, 5) / 100 * 2, stdzr=stdzr)
    ParrayPlotter(m_pa, r_upa.t).plot()
    ParrayPlotter(m_pa, r_upa.z).plot()
    ParrayPlotter(m_pa.t, r_upa.t).plot()
    ParrayPlotter(m_pa.z, r_upa.z).plot()
    ParrayPlotter(m_pa, r_upa, y_scale='standardized').plot()
    ParrayPlotter(m_pa, r_upa, y_scale='transformed').plot()
    ParrayPlotter(m_pa, r_upa, y_scale='transformed', x_scale='standardized').plot()
    ParrayPlotter(m_pa, r_upa, y_scale='standardized', x_scale='transformed').plot()
    ParrayPlotter(m_pa, r_upa, y_scale='transformed', x_scale='standardized').plot()
    ParrayPlotter(m_pa, r_upa.z, y_tick_scale='standardized').plot()
    ParrayPlotter(m_pa, r_upa.t, y_tick_scale='transformed').plot()
    ParrayPlotter(m_pa.z, r_upa, x_tick_scale='standardized').plot()
    ParrayPlotter(m_pa.t, r_upa, x_tick_scale='transformed').plot()


def test_2d_array_plotting():
    stdzr = Standardizer(c={'μ': -5.30, 'σ': 0.582}, d={'μ': -0.307, 'σ': 0.158}, log_vars=['d', 'c'])
    c = np.arange(1, 10, 0.25)
    d = np.arange(1, 10, 0.25)
    c, d = np.meshgrid(c, d)
    Z = np.sin(np.sqrt((c - 5) ** 2 + (d - 5) ** 2))
    mrz = ParameterArray(c=c, d=d, z=Z, stdzr=stdzr)
    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'])
    cs = pp(plt.contourf)
    pp = ParrayPlotter(mrz['c'].t, mrz['d'].t, mrz['z'].t)
    cs = pp(plt.contourf)
    pp = ParrayPlotter(mrz['c'].z, mrz['d'].z, mrz['z'].z)
    cs = pp(plt.contourf)
    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'],
                       x_scale='standardized', y_scale='standardized', z_scale='standardized')
    cs = pp(plt.contourf)
    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'],
                       x_scale='transformed', y_scale='transformed', z_scale='transformed')
    cs = pp(plt.contourf)
    pp = ParrayPlotter(mrz['c'].z, mrz['d'].t, mrz['z'], x_tick_scale='standardized', y_tick_scale='transformed')
    cs = pp(plt.contourf)
    pp = ParrayPlotter(mrz['c'].t, mrz['d'].z, mrz['z'], x_tick_scale='transformed', y_tick_scale='standardized')
    cs = pp(plt.contourf)

def test_colorbar():
    c = np.arange(1, 10, 0.25)
    d = np.arange(1, 10, 0.25)
    c, d = np.meshgrid(c, d)
    Z = np.sin(np.sqrt((c - 5) ** 2 + (d - 5) ** 2))
    stdzr = Standardizer(c={'μ': -5, 'σ': 0.5}, d={'μ': -0.3, 'σ': 0.15}, z={'μ': 2, 'σ': 2}, log_vars=['d', 'c'],
                         logit_vars=['z'])
    mrz = ParameterArray(c=c, d=d, z=Z ** 2, stdzr=stdzr)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'])
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'].t)
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'].z)
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'].t)
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'], z_scale='transformed')
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'].z)
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)

    pp = ParrayPlotter(mrz['c'], mrz['d'], mrz['z'], z_scale='standardized')
    cs = pp(plt.contourf)
    cbar = pp.colorbar(cs)
