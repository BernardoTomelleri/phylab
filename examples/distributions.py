# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:02:40 2021

@author: berni
"""
from phylab import (np, sp, plt, stats, gaussian)
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
        retvals.append(0.5 + np.real(sp.erf(z)/2. + 1j*z**2/np.pi
                                     * mp.hyp2f2(1, 1, 3./2., 2, -z**2)))
    return np.array(retvals)

def three_gaussians(x, loc1=0, sig1=1, sca1=1, loc2=0, sig2=1, sca2=1,
                      loc3=0, sig3=1, sca3=1):
    return gaussian(x, loc1, sig1, sca1) + gaussian(x, loc2, sig2, sca2) + \
           lab.gaussian(x, loc3, sig3, sca3)

def ODR_lorentz(pars, x):
    return pars[1]/np.sqrt(4*(pars[2]*x)**2 + (pars[0]**2 - x**2)**2)

# Data generator and fitter
if gen:
    model = three_gaussians
    dist_pars = [0, 3, 9, 5, 1, 6, 8, 1, 10]
    noise_pars = [0, 1]
    npts = 100
    dy = np.full(npts, noise_pars[1])
    if SEED:
        np.random.seed(SEED)
    noise = np.random.normal(*noise_pars, size=npts)
    data, x = lab.synth_data(model, domain=[-12, 12], pars=dist_pars,
                             noise=noise, npts=npts)
    init = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    pars, covm, deff = lab.propfit(model, x, data, dy=dy, p0=init)
# Sampled resonance curve (Cauchy distribution) fit
else:
    x, data = np.loadtxt('./data/resonance.txt', unpack=True)
    model = lab.lorentzian
    dy = 2
    dx = 1e-2
    init = [4.7, 200, 0.1]
    par_bounds = [[0.0, 5], [0., 500.], [-100, 100]]
    genetic_pars = lab.gen_init(model=model, coords=[x, data],
                                bounds=par_bounds, unc=dy)
    pars, covm, deff = lab.propfit(model, x, data, dx=dx, dy=dy,
                                   p0=genetic_pars, alg='lm')
    popt, pcov, out = lab.ODRfit(ODR_lorentz, x, data, dx=dx, dy=dy, p0=init)
    perr, pcor = lab.errcor(pcov)

# Standard graph with residuals
if tex:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
fig, (axf, axr) = lab.pltfitres(model, x, data, dy=deff, pars=pars)
if gen:
    axf.set_ylabel('$f(x)$')
else:
    axf.set_ylabel('Amplitude [ADC counts]')
    axr.set_xlabel(r'Forcing frequency $\omega$ $\rm{[s^{-1}]}$', x=0.8)
legend = axf.legend(loc='best')
goodness = lab.fit_test(model, coords=[x, data], popt=pars,
                        unc=deff, v=True)

# Plot error distribution for data points
error = (data - model(x, *pars))/deff
figh, axs = plt.subplots(1, 2, constrained_layout=True)
lab.hist_normfit(error, ax=axs[0])
axs[0].set_title('Least-squares error distribution')
axs[0].set_xlabel('deviations/uncertainty')
if gen:
    lab.hist_normfit(noise, ax=axs[1])
    axs[1].set_title('Noise distribution')
else:
    error = np.sqrt((out.delta/dx)**2 + (out.eps/dy)**2)
    lab.hist_normfit(error, ax=axs[1])
    axs[1].set_title('ODR error distribution')
    axs[1].set_xlabel('deviations/uncertainty')
