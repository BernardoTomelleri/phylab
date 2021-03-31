# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 01:51:37 2021

@author: berni
"""
from phylab import (np, plt, grid, curve_fit, stats, DASHED_LINE)
import phylab as lab

''' Variables that control the script '''
naive = False  # Draw confidence bands as model(parameters +/- param. errors)
tex = False  # LaTeX typesetting maths and descriptions

def lin(x, m, q):
    return m*x + q

def perr_bands(ax, model, x, pars, perr=1, nstd=1, fill=False):
    """
    Plots parameter error bands for model by adding and substracting nstd
    standard deviations from optimal pars. Colors in bounded region if fill.
    Warning: Barely somewhat licit with constant/single parameters models

    """
    space = np.linspace(np.min(0.9*x), np.max(1.05*x), 2000)
    pred_up = model(space, *(pars + nstd * perr))
    pred_lo = model(space, *(pars - nstd * perr))
    line_up = ax.plot(space, pred_up, **DASHED_LINE)
    color = line_up[0].get_color()
    line_lo = ax.plot(space, pred_lo, c=color, **DASHED_LINE)
    if fill:
        ax.fill_between(space, pred_lo, pred_up, color=color, alpha=0.3)
    return line_up, line_lo

def conf_delta(x, dfdp, expected, pcov, ci=0.95, covscale=1):
    """
    Generates upper and lower confidence bands for expected model using delta
    method: given vector of partial derivatives of model with respect to
    parameters dfdp and covariance matrix of fitted model parameters pcov.
    Also returns standard deviation from model for prediction band estimates.

    """
    # number of floating parameters from covariance matrix
    p = len(pcov[0]) if not np.isscalar(pcov) else 1
    ndof = len(x) - p
    # convert confidence interval in standard deviations of t distribution
    tval = stats.t.interval(ci, ndof)[-1]

    # Taylor expansion of model around optimal parameters
    df2 = np.zeros(shape=expected.shape)
    for i in range(p):
        for j in range(p):
            df2 += dfdp[i] * dfdp[j] * pcov[i, j]
    model_stdev = np.sqrt(covscale*df2)
    # In a more geometric/Pythonic way, where dfdp.T is a column of gradients
    # written as rows evaluated at every sampled point of the expected model
    # model_stdev = np.array([np.sqrt(grad.T @ pcov @ grad) for grad in dfdp.T])

    # computed deltas over and under best fit model
    delta = tval * model_stdev
    upperband = expected + delta
    lowerband = expected - delta
    return upperband, lowerband, model_stdev

# Input data
x, dx, y, dy = np.loadtxt('./data/ADC.txt', unpack=True)
# Simple linear fit with curve_fit
popt, pcov = curve_fit(lin, x, y, sigma=dy)
perr, pcor = lab.errcor(pcov)
lab.prnpar(popt, perr, model=lin)
chisq, ndof, resn = lab.chitest(lin(x, *popt), y, unc=dy,
                                       ddof=len(popt), v=True)
goodness = lab.fit_test(lin, coords=[x, y], popt=popt, unc=dy, v=True)

lab.rc.typeset(usetex=tex, fontsize=12)
# Graph with residuals and naive 1 sigma confidence bands
fig, (axf, axr) = lab.pltfitres(lin, x, y, dy=dy, pars=popt)
if naive:
    perr_bands(axf, lin, x, pars=popt, perr=perr, fill=True)
# Evaluate actual variance of model with delta method
space = np.linspace(0.9*np.min(x), 1.05*np.max(x), 2000)
mdist = space - np.mean(x)
gradients = np.array([np.ones_like(mdist), mdist])
upb, lob, delta = conf_delta(x, gradients, lin(space, *popt), pcov,
                             cov_scale=1./chisq)
lab.plot_band(axf, space, upb, lob, delta, fill=False, ci=0.95)
axf.set_ylabel('$\Delta V$ [V]')
axr.set_xlabel('ADC reading [digit]', x=0.8)
legend = axf.legend(loc='best')
