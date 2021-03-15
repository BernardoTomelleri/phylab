# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:01:19 2021

@author: berni
"""
import pandas as pd
from phylab import (np, plt, grid, propfit, valid_cov, errcor, prnpar, chitest,
                    curve_fit, tick, logistic, gen_init, R_squared, days_from_epoch,
                    fit_test,)

''' Variables that control the script '''
lin = True # fit data points with linear model
logi = True # fit data points with logistic model
brute = False # brute force search for initial fit parameters

def avg_filter(a, past=7):
    avg = []
    for i in range(len(a)):
        if i > past:
            avg.append(np.mean(a[i : i - past : -1]))
        else: avg.append(None)
    return avg

def linear(x, m, q):
    return m*x + q

def exp_growth(t, tau=1, scale=1):
    return scale*np.exp(t/tau)

NUM_DAYS = 20
url = ('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-'
       'andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
df = pd.read_csv(url, parse_dates=['data'])
last = df[['data', 'nuovi_positivi', 'totale_ospedalizzati',
           'isolamento_domiciliare']].tail(n=NUM_DAYS)
last.insert(2, 'nuovi_positivi (7-day avg.)',
            avg_filter(df['nuovi_positivi'])[-NUM_DAYS:])
patient_data = last[last.columns[1:]].to_numpy(dtype=np.float64).T
days = np.array(range(1, NUM_DAYS + 1))
last.columns = last.columns.str.replace('_', ' ')
fig_fit, axs = plt.subplots(nrows=2, ncols=2)
for patient, ax, name in zip(patient_data, axs.flat, last.columns[1:]):
    pat_err = 1e-2*patient
    # log-lin fitting with exponential
    pars, covm = np.polyfit(days, np.log(patient), deg=1, cov=True)
    pars[0] = 1./pars[0]; pars[1] = np.exp(pars[1])
    errs, corm = errcor(covm)
    prnpar(pars, errs, model=exp_growth)
    chisq, ndof, resn, sigma = chitest(exp_growth(days, *pars), patient,
                                       pat_err, ddof=len(pars), gauss=True, v=True)
    if lin:  # linear fit
        lin_pars, lin_covm, deff = propfit(linear, days, patient, dy=pat_err)
        lin_perr, lin_pcor = errcor(covm)
        prnpar(lin_pars, lin_perr, model=linear)
        lin_chisq, lin_ndof, lin_resn, lin_sigma = \
        chitest(linear(days, *lin_pars), patient, pat_err, ddof=len(lin_pars),
                gauss=True, v=True)
        goodness = fit_test(linear, coords=[days, patient], popt=lin_pars,
                            unc=pat_err, v=True)
    if logi:  # logistic fit
        par_bounds = [[0.0, 1e16], [0., 1.], [-100, 1000]]
        genetic_pars = gen_init(model=logistic, coords=[days, patient],
                                bounds=par_bounds, unc=pat_err)
        popt, pcov, dy = propfit(logistic, days, patient,
                                 p0=genetic_pars, dy=pat_err, alg='trf')
        if brute:
            init = [10, 0.001, 1]
            for i in range(100):
                try:
                    popt, pcov = curve_fit(logistic, days, patient,
                                           p0=init, sigma=pat_err, method='trf')
                except RuntimeError:
                    print('Runtime Error: reached maxfev=800 number of calls')
                if valid_cov(pcov):
                    break
                else: print('Invalid covariance estimate', 'i =', i, name)
                init[0] *= 2; init[1] += 1e-3; init[2] += 1

        perr, pcor = errcor(pcov)
        prnpar(popt, perr, model=logistic)
        chisq_l, ndof_l, resn_l, sigma_l = \
        chitest(logistic(days, *popt), patient, pat_err, ddof=len(popt),
                gauss=True, v=True)
        goodness = fit_test(logistic, coords=[days, patient], popt=popt,
                            unc=pat_err, v=True)

    # Turn last days range into corresponding dates
    from_epoch = days_from_epoch(df['data'].iloc[-1]) - NUM_DAYS
    # Plot results for exponential and logistic fit
    ax.errorbar(days + from_epoch, patient, pat_err, None, marker='.', ms=3,
                elinewidth=1., capsize=1.5, ls='', label=name)
    ax.plot_date(days + from_epoch, exp_growth(days, *pars), ls='-',
                 marker=None, label=r'exp fit $\chi^2 = %.1f/%d$' % (chisq, ndof))
    if logi:
        ax.plot_date(days + from_epoch, logistic(days, *popt), ls='-',
        marker=None, label=r'logistic fit $\chi^2 = %.1f/%d$' % (chisq_l, ndof_l))
    grid(ax, xlab=False, ylab=False)
    if lin:
        ax.plot_date(days + from_epoch, linear(days, *lin_pars), ls='-',
                 marker=None, label=r'linear fit $\chi^2 = %.1f/%d$' % (lin_chisq, lin_ndof))
    #else: logy(ax)
    tick(ax, date=True)
    ax.legend(loc ='best')

fig, axs = plt.subplots(nrows=2, ncols=2)
for ax in axs.flat:
    grid(ax, xlab=False, ylab=False)
    tick(ax, date=True)
last.set_index('data', inplace=True)
last.plot(subplots=True, ax=axs, grid=True, marker='.', lw=1)
