# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:02:40 2021

@author: berni
"""
from phylab import (np, sp, plt)
import mpmath as mp

def voigt_CDF(x, sigma=1, gamma=1):
    """ Voigt cumulative distribution. """
    z = (x + 1j*gamma) / (np.sqrt(2)*sigma)
    return 0.5 + np.real(sp.erf(z)/2. + 1j*z**2/np.pi * mp.hyp2f2(1, 1, 3./2., 2, -z**2))
