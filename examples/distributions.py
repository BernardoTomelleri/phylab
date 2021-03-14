# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:02:40 2021

@author: berni
"""
from phylab import (np, sp, plt)
import phylab as lab
import mpmath as mp
from scipy.stats import norm

def voigt_CDF(x, sigma=1, gamma=1):
    """ Voigt cumulative distribution. """
    retvals = []
    for val in x:
        z = (val + 1j*gamma) / (np.sqrt(2)*sigma)
        retvals.append(0.5 + np.real(sp.erf(z)/2. + 1j*z**2/np.pi * mp.hyp2f2(1, 1, 3./2., 2, -z**2)))
    return np.array(retvals)

def PlotHistNorm(data, log=False):
    # distribution fitting
    param = norm.fit(data)
    mean = param[0]
    sd = param[1]

    # Set generous limits
    xlims = [-5*sd+mean, 5*sd+mean]

    # Plot histogram
    fig, ax = plt.subplots()
    histdata = ax.hist(data,bins=20,alpha=.3,log=log)

    # Generate x points
    x = np.linspace(xlims[0],xlims[1],500)

    # Get y points via Normal PDF with fitted parameters
    pdf_fitted = norm.pdf(x,loc=mean,scale=sd)

    # Get histogram data, in this case bin edges
    xh = [0.5 * (histdata[1][r] + histdata[1][r+1]) for r in range(len(histdata[1])-1)]

    # Get bin width
    binwidth = (max(xh) - min(xh)) / len(histdata[1])

    # Scale the fitted PDF by area of the histogram
    pdf_fitted = pdf_fitted * (len(data) * binwidth)

    #Plot PDF
    ax.plot(x,pdf_fitted,'r--')
    ax.axvline(mean, ymax=max(pdf_fitted)/ax.get_ylim()[1], lw = 1.2, c='b', ls='--', zorder = 1)
    lab.grid(ax, ylab='Entries/bin')
    ax.set_xlim(min(x), max(x))

data, x = lab.synth_data(lab.gaussian, domain=[-5, 5], pars=[0, 1], wnoise=[0, 0.1])
pars, covm, dy = lab.propfit(lab.gaussian, x, data, dy=np.ones_like(data))
fig, ax = plt.subplots()
lab.grid(ax, ylab='$p(x)$')
ax.scatter(x, data)
ax.plot(x, lab.gaussian(x, *pars))
