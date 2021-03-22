# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:19:50 2020

@author: berni
"""
# Standard library imports
import os
import glob
import datetime
from inspect import getfullargspec
from collections import namedtuple

# Scientific library imports
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.odr import odrpack
from scipy import signal as sg
from scipy import special as sp
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import dates

# Configuration file import
from phylab.rc import (
    cm,
    GRID,
    MAJ_TICKS,
    MIN_TICKS,
    DASHED_LINE,
    PLOT_FIT_RESIDUALS,
    MEASURE_MARKER,
    OUTLIER_MARKER,
    NORMRES_MARKER,
    OUTRES_MARKER,
    RES_HLINE,
    SURFACE_PLOT,
    FOURIER_COMPONENTS_PLOT,
    FOURIER_SIGNAL_PLOT,
    FOURIER_LINE,
    FOURIER_MARKER
)

# Standard argument order for periodical functions:
# A: Amplitude, frq: frequency, phs: phase, ofs: DC offset, tau: damp time
args = namedtuple('pars', 'A frq phs ofs')
goodness = namedtuple('ANOVA', 'chi ndof chi_pval R aR F F_pval')

def sine(t, A, frq, phs=0, ofs=0):
    """ Sinusoidal model function. Standard argument order """
    return A*np.sin(2*np.pi*frq*t + phs) + ofs

def cosn(t, A, frq, phs=0, ofs=0):
    """ Cosinusoidal model function. Standard argument order """
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
    x += phs
    if x < T/2.:
        return A*x*((2*x/T) - 1) + ofs
    return A*(3*x - (2*x**2)/T - T) + ofs

def parabola(x_range, A, T, phs=0, ofs=0):
    """ Implementation of series of parabolae from integral of trg(x) """
    y = []
    for x in x_range:
        y.append(modpar((x + phs) % T, A, T, ofs))
    return y

def coope_circ(coords, Xc=0, Yc=0, Rc=1):
    """
    Function of circle pars to be minimized by Coope fit algorithm.
    This is not meant to be used directly.
    """
    x, y = coords
    return Xc*x + Yc*y + Rc

def logistic(x, L=1, k=1, x0=0):
    """ Logistic function or sigmoid curve. """
    return L/(1 + np.exp(k*(x0 - x)))

def lorentzian(x, x0=0, A=1, gamma=0):
    """ Lorentzian resonance curve with center at x0, maximum amplitude A
    and damping coefficient gamma. """
    return A/np.sqrt(4*(gamma*x)**2 + (x0**2 - x**2)**2)

def gaussian(x, mean=0, sigma=1, scale=1):
    """ Gaussian or normal probability distribution. """
    return scale*np.exp( - np.square((x - mean)/(sigma))/2.) / sigma*np.sqrt(2*np.pi)

def gaussian_CDF(x, mean=0, sigma=1, scale=1):
    """ Gaussian or normal cumulative distribution. """
    return scale*(1 + sp.erf((x - mean) / (np.sqrt(2)*sigma)))/2.

def cauchy(x, x0=0, HWHM=1, scale=1):
    """ Cauchy - Lorentz - Breit - Wigner probability distribution. """
    return scale*(HWHM / ((x - x0)**2 + HWHM**2))/np.pi

def cauchy_CDF(x, x0=0, HWHM=1, scale=1):
    """ Cauchy - Lorentz - Breit - Wigner cumulative distribution. """
    return scale*(np.arctan((x - x0) / HWHM)/np.pi + 0.5)

def voigt(x, sigma=1, gamma=1, scale=1):
    """ Voigt profile / probability density function. """
    r2sigma = np.sqrt(2)*sigma
    return scale*np.real(sp.wofz((x + 1j*gamma)/r2sigma)) / r2sigma*np.sqrt(np.pi)

def fder(f, x, pars):
    """ Numerical single variable derivative (for automatic error propagation). """
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
def chitest(prediction, data, unc=1., ddof=0, gauss=False, v=False):
    """
    Evaluates Chi-square goodness of fit test for a function, model, to
    a set of data.

    """
    resn = (data - prediction)/unc
    ndof = len(data) - ddof
    chisq = (resn**2).sum()
    if gauss:
        sigma = (chisq - ndof)/np.sqrt(2*ndof)
        if v:
            print('Chi square/ndof = %.1f/%d [%+.1f]' % (chisq, ndof, sigma))
        return chisq, ndof, resn, sigma
    if v:
        print('Chi square/ndof = %.1f/%d' % (chisq, ndof))
    return chisq, ndof, resn

def chisq(model, x, y, alpha, beta, varnames, pars=None, dy=None):
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
    for idx in constant_pars_indexes:
        ordered_pars[idx] = pars[idx]

    def fxmodel(a, b):
        idxs = np.argwhere(np.invert(fixed))
        ordered_pars[idxs[0]] = a; ordered_pars[idxs[1]] = b
        return model(x, *ordered_pars)
    if dy is None:
        return np.array([[((y - fxmodel(a, b))**2).sum() for a in alpha] for b in beta])
    return np.array([[(((y - fxmodel(a, b))/dy)**2).sum() for a in alpha] for b in beta])

def residual_squares(pars, model, coords, unc=1):
    """
    Sum of squared errors as a function of pars to be minimized by
    differential evolution algorithm. This cannot follow standard
    argument order, therefore is not meant to be used directly.

    """
    x, y = coords
    exval = model(x, *pars)
    return np.sum(((y - exval)/unc) ** 2)

def R_squared(observed, predicted, uncertainty=1):
    """ Returns R square measure of goodness of fit for predicted model. """
    weight = 1./uncertainty
    return 1. - (np.var((observed - predicted)*weight) / np.var(observed*weight))

def adjusted_R(model, coords, popt, unc=1):
    """
    Returns adjusted R squared test for optimal parameters popt calculated
    according to W-MN formula, other forms have different coefficients:
    Wherry/McNemar : (n - 1)/(n - p - 1)
    Wherry : (n - 1)/(n - p)
    Lord : (n + p - 1)/(n - p - 1)
    Stein : (n - 1)/(n - p - 1) * (n - 2)/(n - p - 2) * (n + 1)/n

    """
    x, y = coords
    R = R_squared(y, model(x, *popt), uncertainty=unc)
    n, p = len(y), len(popt)
    coefficient = (n - 1)/(n - p - 1)
    adj = 1 - (1 - R) * coefficient
    return adj, R

def Ftest(model, coords, popt, unc=1):
    """
    One tailed Fisher test for variance of fitted vs constant model.

    """
    x, y = coords
    SSE = residual_squares(popt, model, coords, unc=unc)
    chired = SSE / (len(y) - len(popt))
    SSM = residual_squares(popt, model, [x, np.mean(y)], unc=unc)
    return SSM/len(popt) / chired

def t_test(pars, perr):
    """
    Returns (one-sided) t-statistic values and probabilities for fit
    parameters with standard deviation perr.

    """
    t_vals = pars/perr
    p_vals = stats.t.sf(np.abs(t_vals), df=len(pars))
    return t_vals, p_vals

def fit_test(model, coords, popt, unc=1, v=False):
    """
    Evaluates goodness of fit metrics for best fit parameters popt with model
    to data coords as defined in goodness named tuple.

    Parameters
    ----------
    model : callable
        Model function that minimizes square residuals |y - model(x)|^2.
    coords : array_like
        List or array of [x, y] measured data coordinates.
    popt : array_like
        Optimal values for the parameters of the model.
    unc : array_like, optional
        Uncertainties associated to y data coordinates. The default is 1.

    Returns
    -------
    named_tuple
        Results of goodness-of-fit tests.

    """
    x, y = coords
    ddof = len(popt)
    chisq, ndof, resn = chitest(model(x, *popt), y, unc=unc, ddof=ddof)
    chi_pval = stats.chi2.sf(chisq, ndof)
    adj_R, R = adjusted_R(model, coords, popt)

    SSM = residual_squares(popt, model, [x, np.mean(y)], unc=unc)
    F = SSM/ddof / (chisq/ndof)
    F_pval = stats.f.sf(F, ddof, ndof)

    ANOVA = goodness(chisq, ndof, chi_pval, R, adj_R, F, F_pval)
    if v:
        print(ANOVA)
        #prnpar(pars=ANOVA, perr=np.zeros(len(ANOVA)), manual=ANOVA._fields)
    return ANOVA

def het_cov(A, b, sol, cov):
    """
    HC3, Mackinnon and White heteroscedastic-robust covariance estimator.

    """
    res = b - A @ sol
    leverage = np.diag(A @ (cov @ A.T))
    cov_het = cov @ ((A.T @ A*(res/(1 - leverage))[:, None]**2) @ cov)
    return cov_het

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
            print('corr_%s-%s = %.3f' % (pnms[i], pnms[j], corm[i][j]))

def prnpar(pars, perr=None, model=None, prec=2, manual=None):
    """ Pretty print fit parameters to specified precision %.<prec>f """
    pnms = getfullargspec(model)[0][1:] if manual is None else manual
    if perr is None:
        perr = np.zeros_like(pars)
    for nam, par, err in zip(pnms, pars, perr):
        if err is not None and err > 0:
            dec = np.abs((np.log10(err) - prec)).astype(int)
            print(f'{nam} = {par:.{dec}f} +- {err:.{dec}f}')
        else:
            print(f'{nam} = {par:.{prec}f}')

def RMS(seq):
    """ Evaluates Root Mean Square of data sequence through Numpy module. """
    return np.sqrt(np.mean(np.square(np.abs(seq))))

def RMSE(seq, exp=None):
    """ Associated Root Mean Square Error, expected value = RMS(seq) """
    if exp is None:
        exp = RMS(seq)
    return np.sqrt(np.mean(np.square(seq - exp)))

def abs_devs(seq, around=None):
    """ Evaluates absolute deviations around a central value or expected
    data sequence.

    """
    if around is not None:
        return np.abs(seq - around)
    return np.abs(seq - np.median(seq))

def FWHM(x, y, FT=None):
    """ Evaluates FWHM of fundamental peak for y over dynamic variable x. """
    if FT:
        x = np.fft.fftshift(x); y = np.abs(np.fft.fftshift(y))
    d = y - (np.max(y) / 2.)
    indexes = np.where(d > 0)[0]
    if FT:
        indexes = [i for i in indexes if x[i] > 0]
    if len(indexes) >= 2:
        return np.abs(x[indexes[-1]] - x[indexes[0]])

def optm(x, y, minm=None, absv=None):
    """ Evaluates minima or maxima (default) of y as a function of x. """
    x = np.asarray(x)
    y = np.abs(y) if absv else np.asarray(y)
    if minm:
        yopt = np.min
        idxopt = np.argmin(y)
    else:
        yopt = np.max(y)
        idxopt = np.argmax(y)
    return x[idxopt], yopt

def ldec(seq, upb=None):
    """ Checks if sequence is decreasing, starting from an upper bound. """
    if upb is not None:
        if any(seq[0] > upb):
            return False
        if len(seq) == 1 and all(seq[0] <= upb):
            return True
    check = np.array([seq[k+1] <= seq[k] for k in range(len(seq)-1)])
    if all(bol.all() is True for bol in check):
        return True
    return False

def is_symmetric(a, tol=1e-8):
    "Returns True if array a is symmetric within tolerance tol."
    return np.all(np.abs(a - a.T) < tol)

def valid_cov(covm):
    """
    Checks whether covm is a valid covariance matrix. That is: finite,
    symmetric and not zero on its diagonal.
    """
    if np.isfinite(covm).all() and covm.diagonal().all() != 0. and is_symmetric(covm):
        return True
    return False

def ci2stdevs(dist=stats.norm, ci=0.95, args=None):
    """
    Converts confidence interval to corresponding number of deviations of
    a standard normal distribution (mean = 0, stdev = 1).

    """
    # Conversion to (per)centile point of distribution i.e. percentage
    # of area under the curve up to the right limit of confidence interval.
    pp = (1. + ci) / 2.
    # Percentile point function returns the value of x that contains
    # pp% of the area under normal curve, where CDF(x) = pp
    if args is not None:
        nstd = dist.ppf(pp, *args)
    else:
        nstd = dist.ppf(pp)
    assert nstd == dist.interval(ci)[-1]
    return nstd

def stdevs2ci(dist=stats.norm, sigmas=1, args=None):
    """
    Returns confidence interval/level i.e. area under standard normal
    distribution within n standard deviations/sigmas from the mean.

    """
    if args is not None:
        return 1 - 2*dist.sf(sigmas, *args)
    return 1 - 2*dist.sf(sigmas)

def conf_popt(data, pars, perr, ci=0.95):
    """
    Evaluates ci confidence interval sizes of estimated fit parameters
    around their central values pars with associated uncertainties perr.
    Default confidence probability is 95% ~ 1.96 standard deviations.

    """
    alpha = 1 - ci
    t = stats.t.ppf(1 - alpha/2., df=len(data) - len(pars))
    return t * perr

# LEAST SQUARE FITTING ROUTINES
# Scipy.curve_fit with horizontal error propagation
def propfit(model, xmes, ymes, dx=0., dy=None, p0=None, alg='lm',
            max_iter=20, thr=5, tail=3, rtol=0.5, v=False):
    """
    Modified non-linear least squares (curve_fit) algorithm to fit model to a
    set of data, accounting for x-axis uncertainty using linear error
    propagation onto y-axis uncertainty.

    Parameters
    ----------
    model : callable
        The model function to be fitted to the data.
    xmes : array_like
        Central values of the independent variable where the data is measured.
    ymes : array_like
        Central values of the dependent measured data.
    dy : array_like
        Uncertainties associated to ymes.
    dx : array_like
        Uncertainties associated to xmes.
    p0 : array_like, optional
        Initial guess for the parameters, assumes 1 if p0 is None.
    alg : {‘trf’, ‘dogbox’, ‘lm’}, optional
        Algorithm to perform minimization.
            * 'trf' : Trust Region Reflective algorithm, particularly suitable
            for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : dogleg algorithm with rectangular trust regions,
            typical use case is small problems with bounds. Not recommended
            for problems with rank-deficient Jacobian.
            * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
            Doesn't handle bounds and sparse Jacobians. Usually the most
            efficient method for small unconstrained problems. Default is 'lm'.
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
    popt : ndarray
        Optimal values for the parameters of model that mimimize chi square
        test, found by fitting with propagated uncertainties.
    pcov : 2darray
        The estimated covariance of popt.
    deff : ndarray
        The propagated uncertainty of ymes after the iterated fit proces,
        the same dy passed as argument if propagation was not necessary.

    """
    if np.isscalar(dy):
        dy = np.full(shape=ymes.shape, fill_value=dy)
    deff = dy if dy is not None else np.ones_like(ymes)
    plist = []; elist = []
    for n in range(max_iter):
        popt, pcov = curve_fit(model, xmes, ymes, p0, deff, method=alg,
                               absolute_sigma=False)
        deff = np.sqrt(deff**2 + (dx * fder(model, xmes, popt))**2)
        plist.append(popt)
        elist.append(errcor(pcov)[0])
        con = ldec(np.abs(np.diff(plist[-tail:], axis=0)),
                   upb=elist[-tail]*rtol) if n >= tail else False
        neg = np.mean(deff) > thr*np.mean(dx * abs(fder(model, xmes, popt)))
        if neg or con:
            if v:
                if neg: print(f'x-err negligibility reached in {n} iterations')
                else: print(f'popt values converged in {n} iterations:')
            perr, pcor = errcor(pcov)
            print('Optimal parameters:')
            prnpar(popt, perr, model)
            # Chi square test
            chisq, ndof = chitest(model(xmes, *popt), ymes, deff, ddof=len(popt))[:-1]
            print(f'reduced chi square: {chisq/ndof:.2e}')
            # Normalized parameter covariance
            print('Correlation matrix:\n', pcor)
            break
    if not (neg or con):
        print('Warning: No condition met, number of calls to',
              f'function has reached max_iter = {max_iter}.')
    return popt, pcov, deff

# Scipy.odrpack orthogonal distance regressione
def ODRfit(model, xmes, ymes, dx=0, dy=1, p0=None):
    """
    Finds best-fit model parameters for a set of data using ODR algorithm.
    model function must be in form f(beta[n], x) where beta is an array_like
    containing the n floating paramenters of the model.

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
        Note: by default ODR's covariance matrix is not rescaled cov * res_var
        in other words absolute_sigma is True : the opposite of curve_fit's
        default. Extracting uncertainties from pcov.diagonal() will result in
        perr = out.sd_beta / sqrt(out.res_var)
    out : scipy.odr.odrpack.Output
        The Output class that stores the output of an ODR run, like:
        out.res_var : Residual variance or Reduced chi square
        out.delta : array of deviations along x from predicted model
        out.eps : array of deviations along y from predicted model
        out.iwork[-15] : degrees of freedom of the fit, i.e., the number
        of observations minus the number of parameters actually estimated
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html

    """
    model = odrpack.Model(model)
    data = odrpack.RealData(xmes, ymes, sx=dx, sy=dy)
    odr = odrpack.ODR(data, model, beta0=p0)
    out = odr.run()
    popt = out.beta; pcov = out.cov_beta
    out.pprint()
    print('Chi square/ndof = %.1f/%d' % (out.sum_square, len(ymes)-len(popt)))
    return popt, pcov, out

def gen_init(model, coords, bounds, unc=1):
    """
    Differential evolution algorithm to guess valid initial parameter values
    within bounds in order to fit model to coords.

    """
    if bounds is None:
        bounds = []
        for i in len(getfullargspec(model)[0] - 1):
            bounds.append([-1e8, 1e8])

    result = differential_evolution(residual_squares, bounds,
                                    args=[model, coords, unc], seed=42)
    return result.x

def outlier(model, xmes, ymes, dx=None, dy=None, pars=None, thr=5, mask=False):
    """
    Removes outliers from measured data. A sampled point is considered an
    outlier if it has absolute deviation y - model(x, *pars) > thr*dy.

    """
    if dx is None:
        dx = np.ones_like(xmes)
    if dy is None:
        dy = np.ones_like(ymes)
    if pars is not None:
        isin = [abs_devs(y, around=model(x, *pars)) < thr * sigma
                for x, y, sigma in zip(xmes, ymes, dy)]
    else:
        isin = [abs_devs(y, around=model(x)) < thr * sigma
                for x, y, sigma in zip(xmes, ymes, dy)]
    if mask:
        return xmes[isin], ymes[isin], dx[isin], dy[isin], isin
    return xmes[isin], ymes[isin], dx[isin], dy[isin]

def medianout(data, thr=2.):
    """
    Filters outliers based on median absolute deviation (MAD) around the
    median of measured data.

    """
    devs = abs_devs(data)
    n_sigmas = devs/np.median(devs)
    return data[n_sigmas < thr]

# Circular and elliptical fit functions
def coope(coords, weights=None):
    """
    Attempts to find best fitting circle to array of given coordinates
    (x_i, y_i) prioritizing certain points with weights.
    Returns best-fit center coordinates [x_center, y_center] and radius.
    Adapted from https://ir.canterbury.ac.nz/handle/10092/11104

    """
    npts = len(coords[0]); coords = np.asarray(coords)
    if weights is None:
        weights = np.ones(shape=npts)

    # transformed data arrays for weighted Coope method
    S = np.column_stack((coords.T, np.ones(shape=npts)))
    y = (coords**2).sum(axis=0)
    w = np.diag(weights)

    sol = np.linalg.solve(S.T @ w @ S, S.T @ w @ y)
    center = 0.5*sol[:-1]; radius = np.sqrt(sol[-1] + center.T.dot(center))
    return center, radius

def crcfit(coords, uncerts=None, p0=None):
    """
    Fit circle to a set of coordinates (x_y, y_i) with uncertainties
    (dx_i, dy_i) as weights, using curve_fit with initial parameters p0.

    """
    coords = np.asarray(coords)
    rsq = np.sum(coords**2, axis=0)
    dr = None
    if uncerts is not None:
        uncerts = np.asarray(uncerts)
        dr = np.sum(uncerts**2, axis=0)

    popt, pcov = curve_fit(f=coope_circ, xdata=coords, ydata=rsq, sigma=dr, p0=p0)
    # recover original variables from Coope transformation
    popt[:-1] /= 2.
    popt[-1] = np.sqrt(popt[-1] + (popt[:-1]**2).sum(axis=0))
    pcov[:-1, :-1] /= 2.
    pcov.T[-1] = pcov[-1] = 0.5*np.sqrt(np.abs(pcov[-1]))
    return popt, pcov

def elpfit(coords, uncerts=None):
    """
    Fit ellipse to a set of coordinates (x_y, y_i) with uncertainties
    (dx_i, dy_i) as weights, using NumPy's least square linear algebra solver.
    Returns best fit ellipse parameters as array sol, sum of square residuals
    chisq and estimated parameter covariance pcov.

    """
    x, y = coords
    x = np.atleast_2d(x).T; y = np.atleast_2d(y).T
    A = np.column_stack([x**2, x*y, y**2, x, y])
    b = np.ones_like(y)
    if uncerts is not None and len(uncerts) == len(b):
        A /= uncerts[:, None]
        b /= uncerts[:, None]
    sol, chisq = np.linalg.lstsq(A, b, rcond=None)[:2]
    if chisq:
        ndof = len(y) - len(sol)
        pcov = np.linalg.pinv(A.T @ A)*chisq/ndof
    else:
        print('Warning: covariance of parameters could not be estimated.')
        pcov = None
    return sol, chisq, pcov

def Ell_coords(Xc, Yc, a, b=None, angle=None, arc=1, N=1000):
    """
    Creates N pairs of linearly spaced (x, y) coordinates along an ellipse
    with center coordinates (Xc, Yc); major and minor semiaxes a, b and inclination
    angle (counter-clockwise) between x-axis and major axis. Can create arcs
    of circles by limiting eccentric anomaly between 0 <= t <= arc*2*pi.

    """
    if not b:
        b = a
    elif b > a:
        a, b = b, a
    theta = np.linspace(0, 2*np.pi*arc, num=N)
    if angle:
        x = np.cos(angle)*a*np.cos(theta) - np.sin(angle)*b*np.sin(theta) + Xc
        y = np.sin(angle)*a*np.cos(theta) + np.cos(angle)*b*np.sin(theta) + Yc
    else:
        x = a*np.cos(theta) + Xc
        y = b*np.sin(theta) + Yc
    return x, y

def Ell_imp2std(A, B, C, D, E, F):
    """ Transforms implicit/quadric ellipse parameters into standard ones. """
    DEL = B**2 - 4*A*C; DIS = 2*(A*E**2 + C*D**2 - B*D*E + DEL*F)
    a = -np.sqrt(DIS*((A+C) + np.sqrt((A - C)**2 + B**2)))/DEL
    b = -np.sqrt(DIS*((A+C) - np.sqrt((A - C)**2 + B**2)))/DEL
    Xc = (2*C*D - B*E)/DEL; Yc = (2*A*E - B*D)/DEL
    if B != 0:
        angle = np.arctan(1./B * (C - A - np.sqrt((A - C)**2 + B**2)))
    else:
        angle = 0.5*np.pi if A > C else 0
    return np.asarray([Xc, Yc, a, b, angle])

def Ell_std2imp(Xc, Yc, a, b, angle):
    """ Transforms standard ellipse parameters into implicit/quadric ones. """
    if b > a:
        a, b = b, a
    sin = np.sin(angle); cos = np.cos(angle)
    A = a**2 * sin**2 + b**2 * cos**2
    B = 2*(b**2 - a**2) * sin*cos
    C = a**2 * cos**2 + b**2 * sin**2
    D = -(2*A*Xc + B*Yc)
    E = -(B*Xc + 2*C*Yc)
    F = A*Xc**2 + B*Xc*Yc + C*Yc**2 - (a*b)**2
    return np.array([A, B, C, D, E, F])

def alg_expfit(x, y, dy=None, absolute_sigma=False):
    """
    Weighted algebraic fit for general exponential function
    f(x) = a * exp(b * x) to the data points in arrays xmes and ymes.
    https://mathworld.wolfram.com/LeastSquaresFittingExponential.html

    """
    x, y = np.asarray(x), np.asarray(y)
    S_x2_y, S_y_lny, S_x_y, S_x_y_lny, S_y = np.zeros(shape=5)
    if dy is None:
        dy = np.ones_like(y)
    elif np.isscalar(dy):
        dy = np.full(shape=y.shape, fill_value=dy)
    
    for xi, yi, dyi in zip(x, y, dy):
        wi = 1./dyi**2
        S_x2_y += xi * xi * yi * wi
        S_y_lny += yi * np.log(yi) * wi
        S_x_y += xi * yi
        S_x_y_lny += xi * yi * np.log(yi) * wi
        S_y += yi * wi
    
    Det = S_y * S_x2_y - S_x_y ** 2
    a = (S_x2_y * S_y_lny - S_x_y * S_x_y_lny) / Det
    b = (S_y * S_x_y_lny - S_x_y * S_y_lny) / Det
    popt = [np.exp(a), b]
    
    # Parameter covariance estimates
    chisq, ndof, resn = chitest(popt[0]*np.exp(popt[1]*x), y, unc=dy, ddof=len(popt))
    X = np.column_stack([x/dy.T, 1./dy.T])
    pcov = np.linalg.pinv(X.T @ X)
    if absolute_sigma is False:
         pcov *= chisq/ndof
    return popt, pcov

# UTILITIES FOR MANAGING FIGURE AXES AND OUTPUT GRAPHS
def grid(ax, which='major', xlab=None, ylab=None):
    """
    Adds standard grid and labels for measured data plots to ax.
    Notice: omitting labels results in default x/y [arb. un.]
    to leave a label intentionally blank x/y lab must be set to False.

    """
    ax.grid(which=which, **GRID)
    if xlab is not False:
        ax.set_xlabel('%s' % (xlab if xlab else 'x [arb. un.]'),
                      x=0.9 - len(xlab)/500. if xlab else 0.5)
    if ylab is not False:
        ax.set_ylabel('%s' % (ylab if ylab else 'y [arb. un.]'))
    ax.minorticks_on()
    ax.tick_params(**MAJ_TICKS)
    ax.tick_params(**MIN_TICKS)
    return ax

def tick(ax, xmaj=None, xmin=None, ymaj=None, ymin=None, zmaj=None, zmin=None, date=None):
    """ Adds linearly spaced ticks to ax. """
    if not xmin:
        xmin = xmaj/5. if xmaj else None
    if not ymin:
        ymin = ymaj/5. if xmaj else None
    if xmaj:
        ax.xaxis.set_major_locator(plt.MultipleLocator(xmaj))
    if xmin:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(xmin))
    if ymaj:
        ax.yaxis.set_major_locator(plt.MultipleLocator(ymaj))
    if ymin:
        ax.yaxis.set_minor_locator(plt.MultipleLocator(ymin))
    if zmaj:
        ax.zaxis.set_major_locator(plt.MultipleLocator(zmaj))
    if zmin:
        ax.zaxis.set_minor_locator(plt.MultipleLocator(zmin))
    if date:
        locator = dates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    return ax

def logx(ax, tix=None):
    """ Log-scales x-axis, can add tix logarithmically spaced ticks to ax. """
    ax.set_xscale('log')
    if tix:
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=tix))
        ax.xaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)*.1,
                                                  numticks=tix))
def logy(ax, tix=None):
    """ Log-scales y-axis, can add tix logarithmically spaced ticks to ax. """
    ax.set_yscale('log')
    if tix:
        ax.yaxis.set_major_locator(plt.LogLocator(numticks=tix))
        ax.yaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)*.1,
                                                  numticks=tix))
def logz(ax, tix=None):
    """ Log-scales z-axis, can add tix logarithmically spaced ticks to ax. """
    ax.set_zscale('log')
    if tix:
        ax.zaxis.set_major_locator(plt.LogLocator(numticks=tix))
        ax.zaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)*.1,
                                                  numticks=tix))

def vline_tomax(coords, ax=None, label=None):
    """ Draws vertical line to max. """
    if ax is None:
        ax = plt.gca()
    x, y = coords
    xmax = x[np.argmax(y)]
    ymax=np.max(y)
    ax.axvline(xmax, ymax=ymax/ax.get_ylim()[1], **DASHED_LINE,
               label=f'ymax = {ymax:.2f} in xmax = {xmax:.2f}' if label else None)

def days_from_epoch(date):
    """ Returns number of days passed since UNIX Epoch. """
    return (date - datetime.datetime(1970, 1, 1)).days

def conf_bands(ax, model, x, pars, perr=1, nstd=1, fill=False):
    """
    Plots confidence bands for predicted model within nstd standard deviations
    from optimal parameters pars. Colors in bounded region if fill is True.

    """
    space = np.linspace(np.min(0.9*x), np.max(1.05*x), 2000)
    pred_up = model(space, *(pars + nstd * perr))
    pred_lo = model(space, *(pars - nstd * perr))
    line_up = ax.plot(space, pred_up, ls='--')
    color = line_up[0].get_color()
    line_lo = ax.plot(space, pred_lo, ls='--', c=color)
    if fill:
        ax.fill_between(space, pred_lo, pred_up, color=color, alpha=0.2)
    return line_up, line_lo

def pltfitres(model, xmes, ymes, dx=None, dy=None, pars=None, axs=None, in_out=None, date=None):
    """
    Produces standard plot of best-fit curve describing measured data
    with residuals underneath.

    """
    if axs is None:
        fig, (ax1, ax2) = plt.subplots(**PLOT_FIT_RESIDUALS)
    else:
        ax1, ax2 = axs
    space = np.linspace(np.min(0.9*xmes), np.max(1.05*xmes), 2000)
    chisq, ndof, resn = chitest(model(xmes, *pars), ymes, dy, ddof=len(pars))
    ax1 = grid(ax1, xlab=False, ylab=False)
    if in_out is not None:
        ins, outs = in_out, np.invert(in_out)
        ax1.errorbar(x=xmes[ins], y=ymes[ins], yerr=dy[ins],
                     xerr=None if dx is None else dx[ins], **MEASURE_MARKER)
        ax1.errorbar(x=xmes[outs], y=ymes[outs], yerr=dy[outs],
                     xerr=None if dx is None else dx[outs], **OUTLIER_MARKER)
    else:
        ax1.errorbar(xmes, ymes, dy, dx, **MEASURE_MARKER)
    if date is not None:
        ax1.plot_date(xmes + days_from_epoch(date) - len(xmes), model(xmes, *pars),
                      c='gray', label=r'fit$\chi^2 = %.1f/%d$' % (chisq, ndof))
    else:
        ax1.plot(space, model(space, *pars), c='gray',
                 label=r'fit$\chi^2 = %.1f/%d$' % (chisq, ndof))

    ax2 = grid(ax2, ylab='residuals')
    if in_out is not None:
        ax2.errorbar(xmes[ins], resn[ins], None, None, **NORMRES_MARKER)
        ax2.errorbar(xmes[outs], resn[outs], None, None, **OUTRES_MARKER)
    else:
        ax2.errorbar(xmes, resn, None, None, **NORMRES_MARKER)
    ax2.axhline(0, **RES_HLINE)
    return fig, (ax1, ax2)

def plot3d(x, y, z, xlab=None, ylab=None, zlab=None):
    """ Produces standard 3d plot of z as a function of x and y. """
    X, Y = np.meshgrid(x, y)
    Z = np.atleast_1d(z)
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, **SURFACE_PLOT)
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.jet)
    ax.set_xlabel('%s' % (xlab if xlab else 'x [a.u.]'))
    ax.set_ylabel('%s' % (ylab if ylab else 'y [a.u.]'))
    ax.set_zlabel('%s' % (zlab if zlab else 'z [a.u.]'))
    return fig, ax

def hist_normfit(data, ax=None, log=False):
    """
    Maximum (log) Likelihood Estimation fit for normal/gaussian distribution
    of histogram data.

    """
    pars = stats.norm.fit(data)
    mean, sigma = pars
    prnpar(pars, model=gaussian)
    # Set generous limits
    xlims = [-5*sigma + mean, 5*sigma + mean]
    # Plot histogram
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    histdata = ax.hist(data, bins=20, alpha=0.3, log=log)

    # Get histogram data, in this case bin edges
    xh = [0.5 * (histdata[1][r] + histdata[1][r+1]) for r in range(len(histdata[1]) - 1)]
    binwidth = (max(xh) - min(xh)) / len(histdata[1])
    x = np.linspace(xlims[0], xlims[1], 500)
    pdf_fitted = stats.norm.pdf(x, loc=mean, scale=sigma)
    # Scale the fitted PDF by area of the histogram
    pdf_fitted = pdf_fitted * (len(data)*binwidth)
    #Plot PDF
    ax.plot(x, pdf_fitted, 'r--')
    grid(ax, ylab='Entries/bin')
    return pars

def plotfft(freq, tran, signal=None, norm=False, dB=False, re_im=False, mod_ph=False):
    """
    Plot Fourier transform tran over frequency space freq by itself or
    under its generating signal.

    """
    fft = tran.real if re_im else np.abs(tran)
    if norm:
        fft /= np.max(fft)
    if dB:
        fft = 20*np.log10(np.abs(fft))
    if mod_ph or re_im:
        fig, (ax1, ax2) = plt.subplots(**FOURIER_COMPONENTS_PLOT)
    else:
        fig, (ax2, ax1) = plt.subplots(**FOURIER_SIGNAL_PLOT)
    ax1 = grid(ax1, xlab='Frequency $f$ [Hz]', ylab=False)
    ax1.plot(freq, fft, **FOURIER_LINE)
    ax1.set_ylabel(r'$\widetilde{V}(f)$ Magnitude [%s]' % ('dB' if dB else 'arb. un.'))
    if re_im:
        ax1.set_ylabel('Fourier Transform [Re]')

    ax2 = grid(ax2, xlab='Time $t$ [s]', ylab='$V(t)$ [arb. un.]')
    if mod_ph or re_im:
        fft = tran.imag if re_im else np.angle(tran)
        ax2.plot(freq, fft, **FOURIER_LINE)
        ax1.set_xlabel(None)
        ax2.set_xlabel('Frequency $f$ [Hz]')
        ax2.set_ylabel(r'$\widetilde{V}(f)$ Phase [rad]')
        if re_im:
            ax2.set_ylabel('Fourier Transform [Im]')
    else:
        xmes, ymes, dx, dy = signal
        ax2.errorbar(xmes, ymes, dy, dx, **FOURIER_MARKER)
    if signal:
        ax1, ax2 = ax2, ax1
    return fig, (ax1, ax2)

# UTILITIES FOR FOURIER TRANSFORMS OF DATA ARRAYS
def sampling(space, dev=None, v=False):
    """ Evaluates average sampling interval Dx and its standard deviation. """
    Deltas = np.zeros(len(space)-1)
    sort = np.sort(space)
    for i in range(len(Deltas)):
        Deltas[i] = sort[i+1] - sort[i]
    Dxavg = np.mean(Deltas)
    Dxstd = np.std(Deltas)
    if v:
        print(f"Delta t average = {Dxavg:.2e} s")
        print(f"Delta t stdev = {Dxstd:.2e}")
    if dev:
        return Dxavg, Dxstd
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
    if specres:
        fres = specres
    else:
        fres, frstd = sampling(space=time, dev=True, v=True)
    fftsize = len(time)

    if beta:
        window = lambda M: np.kaiser(M, beta=beta)
    elif window == np.kaiser:
        window = lambda M: np.kaiser(M, beta=0)
    tran = np.fft.rfft(signal*window(len(signal)), fftsize)
    freq = np.fft.rfftfreq(fftsize, d=fres)
    if not specres:
        return freq, tran, fres, frstd
    return freq, tran

# UTILITIES FOR MANAGING DATA AND FILES
def srange(data, x, x_min=0, x_max=1e9):
    """
    Returns sub-array containing data inside selected range over dynamic
    variable x in [x_min, x_max]. If x is equal to data, srange acts as a
    clamp for the data array.

    """
    xup = x[x > x_min]; dup = data[x > x_min]
    sdata = dup[xup < x_max]
    return sdata

def mesrange(x, y, dx=None, dy=None, x_min=0, x_max=1e9):
    """ Restricts measured data to points where x_min < x < x_max. """
    sx = srange(x, x, x_min, x_max)
    sy = srange(y, x, x_min, x_max)
    sdx = None; sdy = None
    if dx is not None:
        sdx = srange(dx, x, x_min, x_max)
    if dy is not None:
        sdy = srange(dy, x, x_min, x_max)
    return sx, sy, sdx, sdy

def uncert(cval, gain=0, read=0):
    """
    Associates uncertainty of measurement to a central value, assuming
    gain/scale and reading error are independent.

    """
    return np.sqrt((cval*gain)**2 + read**2)

def std_unc(measure, ADC=None):
    """ Associates default uncertainty to measured data array."""
    V = np.asarray(measure)
    if ADC:
        return np.ones_like(measure)
    unc = np.diff(V)/2/np.sqrt(12)
    return np.append(unc, np.mean(unc))

def synth_data(model, domain, pars=None, noise=None, npts=100):
    """ Creates noisy data arrays of npts points shaped like model(domain). """
    if len(domain) == 2:
        domain = np.linspace(start=domain[0], stop=domain[1], num=npts)
    if noise is None:
        noise = [0, 1]
    ideal = model(domain, *pars) if pars is not None else model(domain)
    if len(noise) == 2:
        noise = np.random.normal(loc=noise[0], scale=noise[1], size=domain.shape)
    return ideal + noise, domain

def interleave(a, b):
    """ Join two sequences a & b by alternating their elements. """
    merged = [None]*(len(a)+len(b))
    merged[0::2] = a; merged[1::2] = b
    return np.array(merged)

def floop(path=None, usecols=None, v=False):
    """ Allows looping over files of measurements (of a constant value). """
    if path is None:
        path = os.getcwd() + '/data'
    files = glob.glob(path)
    for file in files:
        data = np.loadtxt(file, unpack=True, usecols=usecols)
        if v:
            print(file)
            print('ave = ', np.mean(data))
            print('std = ', np.std(data, ddof=1)/np.sqrt(len(data)))
        yield data
