# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:02:40 2021

@author: berni
"""
from phylab import (np, sp, plt, gaussian_CDF)
import mpmath as mp

def voigt_CDF(x, sigma=1, gamma=1):
    """ Voigt cumulative distribution. """
    retvals = []
    for val in x:
        z = (val + 1j*gamma) / (np.sqrt(2)*sigma)
        retvals.append(0.5 + np.real(sp.erf(z)/2. + 1j*z**2/np.pi * mp.hyp2f2(1, 1, 3./2., 2, -z**2)))
    return np.array(retvals)

def synth_data(model, domain, pars=None, wnoise=None, npts=100):
    if len(domain) == 2:
        domain = np.linspace(start=domain[0], stop=domain[1], num=npts)
    if wnoise is None:
        wnoise = [0, 1]
    ideal = model(domain, *pars) if pars is not None else model(domain) 
    noise = np.random.normal(loc=wnoise[0], scale=wnoise[1], size=domain.shape)
    return ideal + noise, domain

data, x = synth_data(gaussian_CDF, domain=[-1, 1], pars=[0, 1], wnoise=[0, 0.1])
fig, ax = plt.subplots()
ax.plot(x, data)