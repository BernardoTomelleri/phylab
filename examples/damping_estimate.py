# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:08:04 2021

@author: berni
"""
from phylab import (np, plt, grid, errcor, prnpar, chitest)
import phylab as lab

''' Variables that control the script '''
tex = False # LaTeX typesetting maths and descriptions
gen = False # generate dampened oscillation data points

def naive_tau(signal, time):
""" This unfortunately almost never works. Don't use this."""
  idx = np.argmin(np.abs(signal - np.max(signal)/np.e))
  tau = time[idx] - time[np.argmax(signal)]
  return tau
  
def est_tau(signal, time, nbins=50):
    """ Attempts to find rough estimate of signal's damping time tau.
    WARNING: This is very much not guaranteed to work, use at your own risk. """
    AC = signal - np.mean(signal)
    bins = np.array_split(np.asarray(AC), nbins)
    peaks = np.array([np.max(sbin) for sbin in bins])
    dists = peaks - (np.max(peaks)/np.e)
    idx = np.argmin(np.abs(dists))
    bins = np.array_split(np.asarray(time), nbins)
    tau = np.mean(bins[idx]) - np.mean(bins[0])
    return tau

def tau_fit(signal, time, neighbors=5):
    """ Attempts to find signal's damping time tau by linearly fitting the
    positive peaks of the signal with a decaying exponential. """
    peaks = []; ofs = np.mean(signal)
    for idx in range(neighbors, len(signal) - neighbors):
        if signal[idx] == np.max(signal[idx - neighbors : idx + neighbors]) and signal[idx] > ofs:
            peaks.append(idx)
    popt, pcov = np.polyfit(time[peaks], np.log(signal[peaks] - ofs), deg=1, cov=True)
    return popt, pcov, peaks

if gen:
    time = np.linspace(start=0, stop=100, num=1000)
    init = lab.args(A=1, frq=1, phs=0, ofs=0)
    data = lab.dosc(time, *init, tau=30)
    noise = np.random.normal(loc=0, scale=0.1*init.A, size=data.shape)
    t_a = t_b = time; x_a = data; x_b = data + noise
else: t_a, x_a, t_b, x_b = np.loadtxt('./data/beat.txt', unpack = True)
t_min = 1.8; t_max = 130.

# Preliminary plot to visualize the sub-interval of data to analyze
if tex: plt.rc('text', usetex=True); plt.rc('font', family='serif')
fig, axs = plt.subplots(2, 1)
grid(axs[0], xlab='Time [s]', ylab='Pendulum A [ADC counts]')
grid(axs[1], xlab='Time [s]', ylab='Pendulum B [ADC counts]')
axs[0].plot(t_a, x_a, 'gray', alpha=0.6); axs[1].plot(t_b, x_b, 'gray', alpha=0.6)

t_a, x_a, t_b, x_b = lab.mesrange(t_a, x_a, t_b, x_b, t_min, t_max)
axs[0].plot(t_a, x_a, 'k'); axs[1].plot(t_b, x_b, 'k')
dt = np.full(t_a.shape, 2e-5); dx = 2*np.ones_like(x_a)

for time, pos, idx in zip([t_a, t_b], [x_a, x_b], [0, 1]):
    tau = est_tau(pos, time)
    pars, covm, peaks = tau_fit(pos, time)
    tau_f = -1./pars[0]
    ofs = np.mean(pos); amp = np.max(pos) - np.mean(pos)
    axs[idx].plot(time, ofs + amp*np.exp(-time/tau), label=f'est-$\\tau$ = {tau:.2f}')
    axs[idx].plot(time, ofs + amp*np.exp(-time/tau_f), label=f'$\\tau$-fit = {tau_f:.2f}')
    axs[idx].plot(time[peaks], pos[peaks], 'g.')
    axs[idx].legend(loc ='best')
plt.show()