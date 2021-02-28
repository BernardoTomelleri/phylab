# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:19:50 2020

@author: berni
"""
# Standard library imports
import os
import glob
from inspect import getfullargspec
from collections import namedtuple

# Scientific library imports
import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import odrpack
from scipy import signal as sg
from matplotlib import pyplot as plt
from matplotlib import cm

# Standard argument order for periodical functions:
# A: Amplitude, frq: frequency, phs: phase, ofs: DC offset, tau: damp time
args=namedtuple('pars', 'A frq phs ofs')
 
def sine(t, A, frq, phs=0, ofs=0):
    """ Sinusoidal model function. Standard argument order """
    return A*np.sin(2*np.pi*frq*t + phs) + ofs

def cosn(t, A, frq, phs=0, ofs=0):
    return sine(t, A, frq, phs + np.pi/2., ofs)

def dosc(t, A, frq, phs, ofs, tau):
    """ Dampened oscillation model function. Std argument order + tau"""
    return np.exp(-t/tau)*cosn(t, A, frq, phs) + ofs

def trg(t, A, frq, phs=0, ofs=0, duty=0.5):
    """ Triangular wave model function. Std argument order + duty cycle"""
    return A * sg.sawtooth(2*np.pi*frq*t + phs, duty) + ofs
    
def sqw(t, A, frq, phs=0, ofs=0, duty=0.5):
    """ Square wave model function. Std argument order + duty cycle"""
    return A * sg.square(2*np.pi*frq*t + phs, duty) + ofs

def modpar(x, A, T, phs=0, ofs=0):
    """ Modular definition of parabolic signal / integral of a triangle wave."""
    x +=phs
    if (x < T/2.):
        return (A*x*((2*x/T)- 1) + ofs)
    else:
        return (A*(3*x -(2*x**2)/T - T) + ofs)

def parabola(x_range, A, T, phs=0, ofs=0):
    """ Implementation of series of parabolae from integral of trg(x) """
    y = []
    for x in x_range:
        y.append(modpar((x+phs)%T, A, T, ofs))
    return y

def coope_circ(coords, Xc=0, Yc=0, Rc=1):
    x, y = coords
    return Xc*x + Yc*y + Rc

def elps(coords, A, B, C, D, E):
    x, y = coords
    return A*x**2, B*x*y + C*y**2 + D*x + E*y -1

def fder(f, x, pars):
    return np.gradient(f(x, *pars), 1)

def LPF(data, gain=2e-4):
    """ Naive implementation of a digital Low Pass Filter (first order). """
    fltr = [data[0]]
    for x in data[1:]:
        fltr.append(fltr[-1] + gain*(x - fltr[-1]))
    return np.asarray(fltr)

def butf(signal, order, fc, ftype='lp', sampf=None):
    """
    Butterworth {lowpass, highpass, bandpass, bandstop} digital filter

    Parameters
    ----------
    signal : array_like
        Central values of the dependent measured signal over time.
    order : int
        Order of the butterworth filter, the higher the slower the response.
    fc : scalar or tuple
        Critical frequencies of the filter: High, Low cutoff (-3dB) frequencies.
    ftype : str, optional
        Type of filter {'LPF', 'HPF', 'BPF', 'BSF'}. The default is 'lp'.
    sampf : float, optional
        Sampling frequency of the system that measured the signal. If not
        given fc must be normalized to fc/fs. The default is None.

    Returns
    -------
    ndarray
        The digitally filtered signal.

    """
    butw = sg.butter(N=order, Wn=fc, btype=ftype, output='sos', fs=sampf)
    return sg.sosfilt(butw, signal)

# UTILITIES FOR MANAGING PARAMETER ESTIMATES AND TEST RESULTS
# Some functions share a variable v, verbose mode. Activates various print
# statements about variables contained inside the function. 
def chitest(data, unc, model, ddof=0, gauss=False, v=False):
    """ Evaluates Chi-square goodness of fit test for a function, model, to
    a set of data. """
    resn = (data - model)/unc
    ndof = len(data) - ddof
    chisq = (resn**2).sum()
    if gauss:
        sigma = (chisq - ndof)/np.sqrt(2*ndof)
        if v: print('Chi square/ndof = %.1f/%d [%+.1f]' % (chisq, ndof, sigma))
        return chisq, ndof, resn, sigma
    if v: print('Chi square/ndof = %.1f/%d' % (chisq, ndof))
    return chisq, ndof, resn

def chisq(x, y, model, alpha, beta, varnames, pars=None, dy=None):
    """
    Chi-square as a function of two model function parameters (alpha, beta),
    while all others are constrained/constant.

    Parameters
    ----------
    x : array_like
        Central values on which model is evaluated, can be 1d or more.
    y : array_like
        Central values of data fitted by model, can be 1d or more.
    model : callable
        Model function that minimizes square residuals |y - model(x)|^2.
    alpha : array_like
        Array of values the first model parameter can assume, 1-dimensional.
    beta : array_like
        Array of values the second model parameter can assume, 1-dimensional.
    cpars : array_like, optional
        All remaining parameters of model, usually fixed at their optimal
        values, if present. The default is None.
    dy : array_like, optional
        Uncertainties of measured y data, same shape as y. The default is None.

    Returns
    -------
    list
        Sum of weighted square residuals, ready to be plotted as a function
        of alpha and beta. Shape is (len(alpha), len(beta)).

    """
    parnames = getfullargspec(model)[0][1:]
    fixed = [name not in varnames for name in parnames]
    constant_pars_indexes = np.argwhere(fixed)
    ordered_pars = np.zeros(len(parnames))
    for idx in constant_pars_indexes: ordered_pars[idx] = pars[idx]
    def fxmodel(a, b):
        idxs = np.argwhere(np.invert(fixed))
        ordered_pars[idxs[0]] = a; ordered_pars[idxs[1]] = b
        return model(x, *ordered_pars)
    if dy is None:
        return np.array([[((y - fxmodel(a, b))**2).sum() for a in alpha] for b in beta])
    else:
        return np.array([[(((y - fxmodel(a, b))/dy)**2).sum() for a in alpha] for b in beta])

def errcor(covm):
    """ Computes parameter error and correlation matrix from covariance. """
    perr = np.sqrt(covm.diagonal())
    corm = np.array(covm, copy=True)
    for i in range(np.size(corm, 0)):
        for j in range(np.size(corm, 1)):
            corm[i][j] /= perr[i]*perr[j]
    return perr, corm

def prncor(corm, model=None, manual=None):
    """ Pretty print covariance of fit model args to precision %.3f """    
    pnms = getfullargspec(model)[0][1:] if manual is None else manual
    for i in range(np.size(corm, 1)):
        for j in range(1 + i, np.size(corm, 1)):
            print('corr_%s-%s = %.3f' %(pnms[i], pnms[j], corm[i][j]))

def prnpar(pars, perr, model=None, prec=2, manual=None):
    """ Pretty print fit parameters to specified precision %.<prec>f """
    pnms = getfullargspec(model)[0][1:] if manual is None else manual
    dec = np.abs((np.log10(perr) - prec)).astype(int) if all(perr) > 0 else np.ones_like(pars)
    for nam, par, err, d in zip(pnms, pars, perr, dec):
        print(f'{nam} = {par:.{d}f} +- {err:.{d}f}')
        
def RMS(seq):
    """ Evaluates Root Mean Square of data sequence through Numpy module. """
    return np.sqrt(np.mean(np.square(np.abs(seq))))

def RMSE(seq, exp=None):
    """ Associated Root Mean Square Error, expected value = RMS(seq) """
    if not exp: exp = RMS(seq)
    return np.sqrt(np.mean(np.square(seq - exp)))

def abs_devs(seq, around=None):
    """ Evaluates absolute deviations around a central value or expected data sequence."""
    if around is not None: return np.abs(seq - around)
    return np.abs(seq - np.median(seq))

def FWHM(x, y, FT=None):
    """ Evaluates FWHM of fundamental peak for y over dynamic variable x. """
    if FT: x=np.fft.fftshift(x); y=np.abs(np.fft.fftshift(y))
    d = y - (np.max(y) / 2.)
    indexes = np.where(d > 0)[0]
    if FT: indexes = [i for i in indexes if x[i] > 0]
    if len(indexes) >= 2:   
        return np.abs(x[indexes[-1]] - x[indexes[0]])

def optm(x, y, minm=None, absv=None):
    """ Evaluates minima or maxima (default) of y as a function of x. """
    x = np.asarray(x); y = np.abs(y) if absv else np.asarray(y) 
    yopt = np.min(y) if minm else np.max(y)
    xopt = np.where(y == yopt)[0]
    xopt = xopt[0]
    return x[xopt], yopt

def ldec(seq, upb=None):
    """ Checks if sequence is decreasing, starting from an upper bound. """
    if upb is not None: 
        if any(seq[0] > upb): return False
        if len(seq) == 1 and all(seq[0] <= upb): return True
    check = np.array([seq[k+1] <= seq[k] for k in range(len(seq)-1)])
    if all(bol.all() == True for bol in check): return True
    return False

# LEAST SQUARE FITTING ROUTINES
# Scipy.curve_fit with horizontal error propagation 
def propfit(xmes, dx, ymes, dy, model, p0=None, max_iter=20, thr=5, tail=3, tol=0.5, v=False):
    """ Modified non-linear least squares (curve_fit) algorithm to
    fit model to a set of data, accounting for x-axis uncertainty
    using linear error propagation onto y-axis uncertainty.

    Parameters
    ----------
    xmes : array_like
        Central values of the independent variable where the data is measured.
    dx : array_like
        Uncertainties associated to xmes.
    ymes : array_like
        Central values of the dependent measured data.
    dy : array_like
        Uncertainties associated to ymes.
    model : callable
        The model function to be fitted to the data. 
    p0 : array_like, optional
        Initial guess for the parameters, assumes 1 if p0 is None.
    max_iter : int, optional
        Arbitrary natural constant, iteration is stopped if neither condition
        is met before max_iter iterations of the fit routine. 20 by default.
    thr : float, optional
        Arbitrary constant, x-axis uncertainty is considered negligible
        and iterated fit complete whenever dy > thr * |f'(x)|dx. 5 by default.
    tail : int, optional
        Arbitrary natural constant, number of parameter vectors that need
        to converge to the values found in the latest iteration. 3 by default. 
    tol : float, optional
        Arbitrary constant, the fraction of errorbar by which the last tail
        parameters should differ in order to be convergent. 0.5 by default.

    Returns
    -------
    deff : ndarray
        The propagated uncertainty of ymes after the iterated fit proces,
        the same dy passed as argument if propagation was not necessary.
    popt : ndarray
        Optimal values for the parameters of model that mimimize chi square
        test, found by fitting with propagated uncertainties.
    pcov : 2darray
        The estimated covariance of popt.

    """
    deff = np.asarray(dy)
    plist = []; elist = []
    for n in range(max_iter):
        popt, pcov = curve_fit(model, xmes, ymes, p0, deff, 
                               absolute_sigma=False)
        deff = np.sqrt(deff**2 + (dx * fder(model, xmes, popt))**2)
        plist.append(popt); elist.append(errcor(pcov)[0])
        con = ldec(np.abs(np.diff(plist[-tail:], axis=0)), upb=elist[-tail]*tol) if n>=tail else False
        neg = np.mean(deff) > thr*np.mean(dx * abs(fder(model, xmes, popt)))
        # print(n, np.mean(deff) - thr*np.mean(dx * abs(fder(model, xmes,popt)))) DEBUG
        if neg or con:
            if v:
                if neg: print(f'x-err negligibility reached in {n} iterations')
                else: print(f'popt values converged in {n} iterations:')
            perr, pcor = errcor(pcov)
            print('Optimal parameters:')
            prnpar(popt, perr, model)
            # Chi square test
            chisq, ndof, resn = chitest(ymes, deff, model(xmes, *popt),
                                         ddof=len(popt))          
            print(f'Normalized chi square: {chisq/ndof:.2e}')
            # Normalized parameter covariance
            print('Correlation matrix:\n', pcor)
            break
    if v and not(neg or con): print('No condition met, number of calls to',
                                 f'function has reached max_iter = {max_iter}.')
    return popt, pcov, deff

# Scipy.odrpack orthogonal distance regressione 
def ODRfit(xmes, dx, ymes, dy, model, p0=None):
    """ Finds best-fit model parameters for a set of data using ODR algorithm
        (model function must be in form f(beta[n], x)).

    Parameters
    ----------
    xmes : array_like
        Observed data for the independent variable of the regression.
    dx : array_like
        Uncertainties associated to xmes (used as standard deviations).
    ymes : array_like
        array-like, observed data for the dependent variable of the
        regression.
    dy : array_like
        Uncertainties associated to ymes (used as standard deviations).
    model : callable
        The model function that fits the data fcn(beta, x) --> y.
    p0 : array_like, optional
        Reasonable starting values for the fit parameters.
    
    Returns
    -------
    popt : ndarray
        Optimal values for the parameters (beta) of the model
    pcov : 2darray
        The estimated covariance of popt.
        
    """
    model = odrpack.Model(model)
    data = odrpack.RealData(xmes, ymes, sx=dx, sy=dy)
    odr = odrpack.ODR(data, model, beta0=p0)
    out = odr.run()
    popt = out.beta; pcov = out.cov_beta
    out.pprint()
    print('Chi square/ndof = %.1f/%d' % (out.sum_square, len(ymes)-len(p0)))
    return popt, pcov

def outlier(xmes, dx, ymes, dy, model, pars, thr=5, out=False):
    """ Removes outliers from measured data. A sampled point is considered an
    outlier if it has absolute deviation y - model(x, *pars) > thr*dy. """
    isin = [abs_devs(y, around=model(x, *pars)) < thr*sigma
            for x, y, sigma in zip(xmes, ymes, dy)]
    if out:
        isout = np.invert(isin)
        return (xmes[isin], dx[isin], ymes[isin], dy[isin], 
            xmes[isout], dx[isout], ymes[isout], dy[isout])
    
    return xmes[isin], dx[isin], ymes[isin], dy[isin]

def medianout(data, thr=2.):
    """ Filters outliers based on median absolute deviation (MAD) around the
    median of measured data. """
    devs = abs_devs(data)
    n_sigmas = devs/np.median(devs)
    return data[n_sigmas < thr]

# Circular and elliptical fit functions    
def coope(coords, weights=None):
    npts = len(coords[0]); coords = np.asarray(coords)
    if weights is not None and len(weights) != npts: raise Exception
    else: weights = np.ones(shape = npts)
    
    # transformed data arrays for weighted Coope method
    S = np.column_stack((coords.T, np.ones(shape = npts)))
    y = (coords**2).sum(axis=0)
    w = np.diag(weights)
    
    sol = np.linalg.solve(S.T @ w @ S, S.T @ w @ y)
    center = 0.5*sol[:-1]; radius = np.sqrt(sol[-1] + center.T.dot(center))
    return center, radius

def crcfit(coords, uncerts=None, p0=None):
    rsq = (coords**2).sum(axis=0)
    dr = (uncerts**2).sum(axis=0) if uncerts else None
    
    popt, pcov = curve_fit(f=coope_circ, xdata=coords, ydata=rsq, sigma=dr, p0=p0)
    # recover original variables from Coope transformation
    popt[:-1]/=2.
    popt[-1] = np.sqrt(popt[-1] + (popt[:-1]**2).sum(axis=0))
    pcov[:-1, :-1]/=2.
    pcov.T[-1] = pcov[-1] = 0.5*np.sqrt(np.abs(pcov[-1]))
    return popt, pcov

def elpfit(coords, uncerts=None):
    x, y = coords; x = np.atleast_2d(x).T; y = np.atleast_2d(y).T
    A = np.column_stack([x**2, x*y, y**2, x, y])
    b = np.ones_like(y)
    if uncerts is not None and len(uncerts) == len(b): A/=uncerts[:,None]; b/=uncerts[:,None]
    sol, chisq = np.linalg.lstsq(A, b, rcond=None)[:2]
    if chisq: 
        pcov = np.linalg.pinv(A.T @ A)*chisq/len(b)
    else:
        print('Covariance of parameters could not be estimated')
        pcov=None
    return sol, chisq, pcov

def Ell_coords(Xc, Yc, a, b=None, angle=None, arc=1, N=1000):
    """ Creates N pairs of linearly spaced (x, y) coordinates along an ellipse
    with center coordinates (Xc, Yc); major and minor semiaxes a, b and inclination
    angle (counter-clockwise) between x-axis and major axis. Can create arcs
    of circles by limiting eccentric anomaly between 0 <= t <= arc*2*pi."""
    if not b: b = a
    elif b > a: a, b = b, a
    theta = np.linspace(0, 2*np.pi*arc, num=N)
    if angle:
        x = np.cos(angle)*a*np.cos(theta) - np.sin(angle)*b*np.sin(theta) + Xc
        y = np.sin(angle)*a*np.cos(theta) + np.cos(angle)*b*np.sin(theta) + Yc
    else:
        x = a*np.cos(theta) + Xc
        y = b*np.sin(theta) + Yc
    return x, y
    
def Ell_imp2std(A, B, C, D, E, F):
    DEL = B**2 - 4*A*C; DIS = 2*(A*E**2 + C*D**2 - B*D*E + DEL*F)
    a = -np.sqrt(DIS*((A+C) + np.sqrt((A - C)**2 + B**2)))/DEL
    b = -np.sqrt(DIS*((A+C) - np.sqrt((A - C)**2 + B**2)))/DEL
    Xc = (2*C*D - B*E)/DEL; Yc = (2*A*E - B*D)/DEL
    if B != 0: tilt = np.arctan(1./B *(C - A - np.sqrt((A - C)**2 + B**2)))
    else: tilt = 0.5*np.pi if A > C else 0 
    return np.asarray([Xc, Yc, a, b, tilt])

def Ell_std2imp(Xc, Yc, a, b, angle):
    if b > a: a, b = b, a
    sin = np.sin(angle); cos = np.cos(angle)
    A = a**2 * sin**2 + b**2 * cos**2  
    B = 2*(b**2 - a**2) * sin*cos
    C = a**2 * cos**2 + b**2 * sin**2
    D = -(2*A*Xc + B*Yc)
    E = -(B*Xc + 2*C*Yc)
    F = A*Xc**2 + B*Xc*Yc + C*Yc**2 - (a*b)**2
    return np.array([A, B, C, D, E, F])

# UTILITIES FOR MANAGING FIGURE AXES AND OUTPUT GRAPHS
def grid(ax, xlab = None, ylab = None):
    """ Adds standard grid and labels for measured data plots to ax.
    Notice: omitting labels results in default x/y [arb. un.]. To leave a
    label intentionally blank x/y lab must be set to False. """
    ax.grid(color = 'gray', ls = '--', alpha=0.7)
    if xlab is not False:
        ax.set_xlabel('%s' %(xlab if xlab else 'x [arb. un.]'),
                      x=0.9 - len(xlab)/500 if xlab else 0.9)
    if ylab is not False:
        ax.set_ylabel('%s' %(ylab if ylab else 'y [arb. un.]' ))
    ax.minorticks_on()
    ax.tick_params(direction='in', length=4, width=1., top=True, right=True)
    ax.tick_params(which='minor', direction='in', width=1., top=True, right=True)
    return ax
    
def tick(ax, xmaj=None, xmin=None, ymaj=None, ymin=None, zmaj=None, zmin=None):
    """ Adds linearly spaced ticks to ax. """
    if not xmin: xmin = xmaj/5. if xmaj else None
    if not ymin: ymin = ymaj/5. if xmaj else None
    if xmaj: ax.xaxis.set_major_locator(plt.MultipleLocator(xmaj))
    if xmin: ax.xaxis.set_minor_locator(plt.MultipleLocator(xmin))
    if ymaj: ax.yaxis.set_major_locator(plt.MultipleLocator(ymaj))
    if ymin: ax.yaxis.set_minor_locator(plt.MultipleLocator(ymin))
    if zmaj: ax.zaxis.set_major_locator(plt.MultipleLocator(zmaj))
    if zmin: ax.zaxis.set_minor_locator(plt.MultipleLocator(zmin))
    return ax

def logx(ax, tix=None):
    """ Log-scales x-axis, can add tix logarithmically spaced ticks to ax. """
    ax.set_xscale('log')
    if tix:
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=tix))
        ax.xaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)*.1,
                                                  numticks = tix))
def logy(ax, tix=None):
    """ Log-scales y-axis, can add tix logarithmically spaced ticks to ax. """
    ax.set_yscale('log')
    if tix:
        ax.yaxis.set_major_locator(plt.LogLocator(numticks=tix))
        ax.yaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)*.1,
                                                  numticks = tix))
def logz(ax, tix=None):
    """ Log-scales z-axis, can add tix logarithmically spaced ticks to ax. """
    ax.set_zscale('log')
    if tix:
        ax.zaxis.set_major_locator(plt.LogLocator(numticks=tix))
        ax.zaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)*.1,
                                                  numticks = tix))
def pltfitres(xmes, dx, ymes, dy=None, model=None, pars=None, out=None):
# Variables that control the script 
    # kwargs.setdefault(
    #     {
    #     'log' : True, # log-scale axis/es
    #     'dB' : True, # plots response y-axis in deciBels
    #     'tex' : True, # LaTeX typesetting maths and descriptions
    #     })
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={
    'wspace':0.05, 'hspace':0.05, 'height_ratios': [3, 1]})
    space = np.linspace(np.min(0.9*xmes), np.max(1.05*xmes), 2000)
    if out is not None: space = np.linspace(np.min(0.9*out), np.max(1.05*out), 2000)
    chisq, ndof, resn = chitest(ymes, dy, model(xmes, *pars), ddof=len(pars))
    ax1 = grid(ax1, xlab = False, ylab = False)
    ax1.errorbar(xmes, ymes, dy, dx, 'ko', ms=1.5, elinewidth=1., capsize=1.5,
             ls='', label='data')
    ax1.plot(space, model(space, *pars), c='gray', 
             label='fit$\chi^2 = %.1f/%d$' %(chisq, ndof))
    
    ax2 = grid(ax2, xlab = False, ylab = 'residuals')
    ax2.errorbar(xmes, resn, None , None, 'ko', elinewidth=0.5, capsize=1.,
             ms=1., ls='--', zorder=5)
    ax2.axhline(0, c='r', alpha=0.7, zorder=10)
    return fig, (ax1, ax2)

def plot3d(x, y, z, xlab=None, ylab=None, zlab=None):
    X, Y = np.meshgrid(x, y); Z=np.atleast_1d(z)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                       linewidth=0, antialiased=False, alpha=0.6)
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.jet)
    ax.set_xlabel('%s' %(xlab if xlab else 'x [a.u.]'))
    ax.set_ylabel('%s' %(ylab if ylab else 'y [a.u.]'))
    ax.set_zlabel('%s' %(zlab if zlab else 'z [a.u.]'))
    return fig, ax

def plotfft(freq, tran, signal=None, norm=False, dB=False, re_im=False, mod_ph=False):
    fft = tran.real if re_im  else np.abs(tran)
    if norm: fft/=np.max(fft)
    if dB: fft = 20*np.log10(np.abs(fft))
    if mod_ph or re_im:
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True,
                                       gridspec_kw={'wspace':0.05, 'hspace':0.05})
    else: fig, (ax2, ax1) = plt.subplots(2, 1,
                                         gridspec_kw={'wspace':0.25, 'hspace':0.25}) 
    ax1 = grid(ax1, xlab = 'Frequency $f$ [Hz]', ylab = False)
    ax1.plot(freq, fft, c='k', lw='0.9')
    ax1.set_ylabel('$\widetilde{V}(f)$ Magnitude [%s]' %('dB' if dB else 'arb. un.'))
    if re_im: ax1.set_ylabel('Fourier Transform [Re]')    

    ax2 = grid(ax2, xlab = 'Time $t$ [s]', ylab = '$V(t)$ [arb. un.]')
    if mod_ph or re_im: 
        fft = tran.imag if re_im else np.angle(tran)
        ax2.plot(freq, fft, c='k', lw='0.9')
        ax2.set_xlabel('Frequency $f$ [Hz]'); ax1.set_xlabel(None) 
        ax2.set_ylabel('$\widetilde{V}(f)$ Phase [rad]')
        if re_im: ax2.set_ylabel('Fourier Transform [Im]')    
    else:
        xmes, dx, ymes, dy = signal
        #ax2.plot(xmes, ymes, 'ko', ms=0.5, ls='-', lw=0.7, label='data')
        ax2.errorbar(xmes, ymes, dy, dx, 'ko', ms=1.2, elinewidth=0.8,
                     capsize= 1.1, ls='', lw=0.7, label='data', zorder=5)
    if signal: ax1, ax2 = [ax2, ax1]
    return fig, (ax1, ax2)

# UTILITIES FOR FOURIER TRANSFORMS OF DATA ARRAYS
def sampling(space, dev=None, v=False):
    """ Evaluates average sampling interval Dx and its standard deviation. """ 
    Deltas=np.zeros(len(space)-1)
    sort=np.sort(space)
    for i in range(len(Deltas)):
        Deltas[i] = sort[i+1] - sort[i]
    Dxavg=np.mean(Deltas)
    Dxstd=np.std(Deltas)
    if v:
        print(f"Delta t average = {Dxavg:.2e} s" )
        print(f"Delta t stdev = {Dxstd:.2e}")
    if dev: return Dxavg, Dxstd
    return Dxavg

def FFT(time, signal, window=None, beta=0, specres=None):
    """
    Computes Discrete Fourier Transform of signal and its frequency space.

    Parameters
    ----------
    time : array_like
        Time interval over which the signal is sampled.
    signal : array_like
        The ADC sampled signal (real/complex) to be transformed with fft.
    window : function, optional
        Numpy window function with which to filter fft. The default is None.
    beta : int, optional
        Kaiser window's beta parameter. The default is 0, rect filter.
    specres : float, optional
        Sample spacing of the system that acquired signal. The default is None.

    Returns
    -------
    freq : ndarray
        Discrete sample frequency spectrum.
    tran : complex ndarray
        One-dimensional discrete Fourier Transform of the signal.

    """
    # Spectral resolution and number of points over which FFT is computed      
    if specres: fres = specres
    else: fres, frstd = sampling(space=time, dev=True, v=True)
    fftsize = len(time)
    
    if beta: window = lambda M: np.kaiser(M, beta=beta)
    elif window == np.kaiser: window = lambda M: np.kaiser(M, beta=0)
    tran = np.fft.rfft(signal*window(len(signal)), fftsize)
    freq = np.fft.rfftfreq(fftsize, d=fres)
    if not specres: return freq, tran, fres, frstd
    return freq, tran
    
# UTILITIES FOR MANAGING DATA FILES
def srange(data, x, x_min=0, x_max=1e9):
    """ Returns sub-array containing data inside selected range over 
        dynamic variable x in [x_min, x_max]. If x is equal to data,
        srange acts as a clamp for the data array"""
    xup = x[x > x_min]; dup = data[x > x_min]
    sdata = dup[xup < x_max]
    return sdata
    
def mesrange(x, dx, y, dy, x_min=0, x_max=1e9):
    """ Restricts measured data to points where x_min < x < x_max. """ 
    sx = srange(x, x, x_min, x_max); sdx = srange(dx, x, x_min, x_max)
    sy = srange(y, x, x_min, x_max); sdy = srange(dy, x, x_min, x_max)
    return sx, sdx, sy, sdy

def uncert(cval, gain=0, read=0):
    """ Associates uncertainty of measurement to a central value, assuming
    gain/scale and reading error are independent. """
    return np.sqrt((cval*gain)**2 + read**2)

def std_unc(measure, ADC=None):
    """ Associates default uncertainty to measured data array."""
    V = np.asarray(measure)
    if ADC: return np.ones_like(measure)
    unc = np.diff(V)/2/np.sqrt(12)
    return np.append(unc, np.mean(unc))

def interleave(a, b, inv=False):
    """ Join two sequences a & b by alternating their elements. """
    merged = [None]*(len(a)+len(b))
    merged[0::2] = a; merged[1::2] = b
    return np.array(merged)
    
def floop(path=None, usecols=None, v=False):
    """ Allows looping over files of measurements (of a constant value). """
    if path is None: path = os.getcwd() + '/data'
    files = glob.glob(path)
    for file in files:
        data = np.loadtxt(file, unpack = True, usecols=usecols)
        if v:
            print(file)
            print('ave = ', np.mean(data))
            print('std = ', np.std(data, ddof=1)/np.sqrt(len(data)))
        yield data
