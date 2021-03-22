# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 01:51:37 2021

@author: berni
"""
from phylab import (np, plt, grid, curve_fit)
import phylab as lab

''' Variables that control the script '''
proper = True  # Evaluate actual variance of model with Taylor expansion

def lin(x, m, q):
    return m*x + q

def confidence_band(x, dfdp, predict, pcov, ci=0.95, covscale=1):
    from scipy.stats import t
    # Given the confidence probability ci = 100(1 - alpha)
    # alpha = 1 - ci, tail = 1 - alpha/2 = (1 + ci)/2
    prb = (1.0 + ci)/2.
    ndof = len(x) - len(popt)
    tval = t.ppf(prb, ndof)
    # Number of parameters from covariance matrix
    n = len(pcov[0])
    df2 = np.zeros(shape=x.shape)
    for j in range(n):
        for k in range(n):
            df2 += dfdp[j] * dfdp[k] * pcov[j,k]
    df = np.sqrt(covscale*df2)
    delta = tval * df
    upperband = predict + delta
    lowerband = predict - delta
    return upperband, lowerband

# Input data
x, dx, y, dy = np.loadtxt('./data/ADC.txt', unpack=True)
# Simple linear fit with curve_fit
popt, pcov = curve_fit(lin, x, y, sigma=dy)
perr, pcor = lab.errcor(pcov)
lab.prnpar(popt, perr, model=lin)
chisq, ndof, resn = lab.chitest(lin(x, *popt), y, unc=dy,
                                       ddof=len(popt), v=True)
goodness = lab.fit_test(lin, coords=[x, y], popt=popt, unc=dy, v=True)

# Graph with residuals and naive 1 sigma confidence bands
fig, (axf, axr) = lab.pltfitres(lin, x, y, dy=dy, pars=popt)
axf.set_ylabel('$\Delta V$ [V]')
lab.conf_bands(axf, lin, x, popt, perr=perr, fill=True)
axr.set_xlabel('x [digit]', x=0.9)

if proper:
    upb, lob = confidence_band(x, [1, x], lin(x, *popt), pcov, ci=0.683)
    axf.plot(x, upb, c='r', ls='--', lw=0.8,
             zorder=10, alpha=0.7, label=r'95 \% confidence band')
    axf.plot(x, lob, c='r', ls='--', lw=0.8,
             zorder=10, alpha=0.7)
legend = axf.legend(loc='best')