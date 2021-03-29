# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:05:34 2021

Phylab configuration file

@author: berni
"""
from matplotlib import cm, rcParams

def typeset(usetex=True, preamble=True, fontsize=12):
    """
    Sets up LaTeX typesetting for matplotlib output graphs.
    Requires a functioning LaTeX installation on your machine.

    """
    rcParams['text.usetex'] = usetex
    rcParams['font.size'] = fontsize
    rcParams['font.family'] = 'serif' if usetex else 'sans-serif'

    if preamble:
        rcParams['text.latex.preamble'] = r'''
            \usepackage{amsmath}
            \usepackage{siunitx}
            \usepackage{booktabs}
            '''

# By default TeX rendering is disabled, to enable call typeset inside module.
typeset(usetex=False, preamble=False, fontsize=12)

# By default verbose print statements inside functions are disabled,
# set VERBOSE to True to modify global behaviour
VERBOSE = False

# Errorbar and Line2D properties for plots in order of appearance in lab
GRID = {
    'color' : 'gray',
    'linestyle' : '--',
    'alpha' : 0.7
}

MAJ_TICKS = {
    'direction' : 'in',
    'length' : 4,
    'width' : 1,
    'top' : True,
    'right' : True
}

MIN_TICKS = {
    'which' : 'minor',
    'direction' : 'in',
    'width' : 1,
    'top' : True,
    'right' : True
}

DASHED_LINE = {
    'linestyle' : '--',
    'linewidth' : 1.2
}

PLOT_FIT_RESIDUALS = {
    'nrows' : 2,
    'ncols' : 1,
    'sharex' : True,
    'gridspec_kw' : {'height_ratios' : [3, 1]},
    'constrained_layout' : True
}

MEASURE_MARKER = {
    'color' : 'black',
    'marker' : 'o',
    'markersize' : 2,
    'elinewidth' : 1,
    'capsize' : 1.5,
    'linestyle' : '',
    'linewidth' : 1,
    'label' : 'data'
}

OUTLIER_MARKER = {
    'color' : 'green',
    'marker' : 'x',
    'markersize' : 3,
    'elinewidth' : 1,
    'capsize' : 1.5,
    'linestyle' : '',
    'label' : 'outlier',
    'zorder' : 5
}

NORMRES_MARKER = {
    'color' : 'black',
    'marker' : 'o',
    'markersize' : 2,
    'elinewidth' : 0.5,
    'capsize' : 1,
    'linestyle' : '--',
    'linewidth' : 1,
    'zorder' : 5
}

OUTRES_MARKER = {
    'color' : 'green',
    'marker' : 'x',
    'markersize' : 3,
    'elinewidth' : 0.5,
    'capsize' : 1,
    'linestyle' : '--',
    'linewidth' : 1
}

RES_HLINE = {
    'color' : 'red',
    'alpha' : 0.7,
    'zorder' : 10
}

SURFACE_PLOT = {
    'rstride' : 1,
    'cstride' : 1,
    'cmap' : cm.jet,
    'linewidth' : 0,
    'antialiased' : False,
    'alpha' : 0.6
}

FOURIER_COMPONENTS_PLOT = {
    'nrows' : 2,
    'ncols' : 1,
    'sharex' : True,
    'gridspec_kw' : {'wspace': 0.05, 'hspace': 0.05},
    'constrained_layout' : True
}

FOURIER_SIGNAL_PLOT = {
    'nrows' : 2,
    'ncols' : 1,
    'gridspec_kw' : {'wspace': 0.25, 'hspace': 0.25},
    'constrained_layout' : True
}

FOURIER_LINE = {
    'color' :'black',
    'linewidth' : 0.9
}

FOURIER_MARKER = {
    'color' : 'black',
    'marker' : 'o',
    'markersize' : 1.2,
    'elinewidth' : 0.8,
    'capsize' : 1.1,
    'linestyle' : '',
    'linewidth' : 0.7,
    'label' : 'data'
}

LEGEND = {
    'loc' : 'best',
    'framealpha' : 0.7
}

TICK_LABEL = {
    'useOffset' : False,
    'useMathText' : True
}

"""
Notes
-----
Unpacking these dictionaries (with **dict) inside pyplot functions is
identically equivalent to setting the same keyword arguments manually.

>>> from phylab import (plt, grid)
>>> fig, ax = plt.subplots()

calling
>>> ax.tick_params(which='minor', direction='in', width=1, top=True, right=True)
is equivalent to
>>> ax.tick_params(**MIN_TICKS)

But modifying the default behaviour of phylab's functions is much easier and
less error-prone in the second case.

"""
