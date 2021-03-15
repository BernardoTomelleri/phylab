# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:02:40 2021

@author: berni
"""
from phylab import (np, sp, plt, stats)
import phylab as lab
import mpmath as mp

''' Variables that control the script '''
tex = False  # LaTeX typesetting maths and descriptions
gen = False  # generate distribution data points
SEED = 3  # for repeatable results. Change to None to disable

def voigt_CDF(x, sigma=1, gamma=1):
    """ Voigt cumulative distribution. """
    retvals = []
    for val in x:
        z = (val + 1j*gamma) / (np.sqrt(2)*sigma)
        retvals.append(0.5 + np.real(sp.erf(z)/2. + 1j*z**2/np.pi * mp.hyp2f2(1, 1, 3./2., 2, -z**2)))
    return np.array(retvals)

def hist_normfit(data, log=False):
    # distribution fitting
    param = stats.norm.fit(data)
    mean = param[0]; sd = param[1]
    # Set generous limits
    xlims = [-5*sd+mean, 5*sd+mean]
    # Plot histogram
    fig, ax = plt.subplots()
    histdata = ax.hist(data,bins=20,alpha=.3,log=log)
    # Get histogram data, in this case bin edges
    xh = [0.5 * (histdata[1][r] + histdata[1][r+1]) for r in range(len(histdata[1])-1)]
    binwidth = (max(xh) - min(xh)) / len(histdata[1])

    x = np.linspace(xlims[0],xlims[1],500)
    pdf_fitted = stats.norm.pdf(x,loc=mean,scale=sd)
    # Scale the fitted PDF by area of the histogram
    pdf_fitted = pdf_fitted * (len(data) * binwidth)
    #Plot PDF
    ax.plot(x,pdf_fitted,'r--')
    ax.axvline(mean, ymax=max(pdf_fitted)/ax.get_ylim()[1], lw = 1.2, c='b',
               ls='--', zorder = 1)
    lab.grid(ax, ylab='Entries/bin')
    ax.set_title('Error distribution')

def three_gaussians(x, loc1=0, sig1=1, sca1=1, loc2=0, sig2=1, sca2=1,
                      loc3=0, sig3=1, sca3=1):
    return lab.gaussian(x, loc1, sig1, sca1) + \
     lab.gaussian(x, loc2, sig2, sca2) + lab.gaussian(x, loc3, sig3, sca3)

dist_pars = [0, 3, 9, 5, 1, 6, 8, 1, 10]
noise_pars = [0, 1]
npts = 100
dy = np.full(npts, noise_pars[1])
# Data generator and fitter
if SEED: 
    np.random.seed(SEED)
data, x = lab.synth_data(three_gaussians, domain=[-12, 12], pars=dist_pars,
                         wnoise=noise_pars, npts=npts)
init = [1, 1, 1, 1, 1, 1, 1, 1, 1]
pars, covm, deff = lab.propfit(three_gaussians, x, data, dy=dy, p0=init)
# Standard graph with residuals
if tex:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
fig, (axf, axr) = lab.pltfitres(three_gaussians, x, data, dy=deff, pars=pars)
axf.set_ylabel('$f(x)$')
legend = axf.legend(loc='best')
goodness = lab.fit_test(three_gaussians, coords=[x, data], popt=pars,
                        unc=deff, v=True)
axf.plot(x, lab.gaussian(x, *pars[:3]), ls='--')
axf.plot(x, lab.gaussian(x, *pars[3:6]), ls='--')
axf.plot(x, lab.gaussian(x, *pars[-3:]), ls='--')

# Plot error distribution for data points
if SEED:
    np.random.seed(SEED)
hist_normfit(np.random.normal(*noise_pars, size=npts))
