# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:02:40 2021

@author: berni
"""
from phylab import (np, sp, plt)
import phylab as lab
import mpmath as mp

def voigt_CDF(x, sigma=1, gamma=1):
    """ Voigt cumulative distribution. """
    retvals = []
    for val in x:
        z = (val + 1j*gamma) / (np.sqrt(2)*sigma)
        retvals.append(0.5 + np.real(sp.erf(z)/2. + 1j*z**2/np.pi * mp.hyp2f2(1, 1, 3./2., 2, -z**2)))
    return np.array(retvals)

data, x = lab.synth_data(voigt_CDF, domain=[-1, 1], pars=[1, 1], wnoise=[0, 0.1])
fig, ax = plt.subplots()
lab.grid(ax, ylab='$p(x)$')
ax.plot(x, data)