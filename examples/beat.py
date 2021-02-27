# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:57:02 2021

@author: berni
"""
from phylab.lab import (np, plt, grid, propfit, errcor, prnpar, chitest, pltfitres)
from phylab import lab

''' Variables that control the script '''
tix = True # manually choose spacing between axis ticks
tex = True # LaTeX typesetting maths and descriptions

def beat(t, A, frq_d, frq_m, phs_f=0, phs_c=0, ofs=0, tau=1):
    return np.exp(-t/tau)*2*A*(np.cos(np.pi*frq_d*t + (phs_f + phs_c)/2.)
                             * np.cos(np.pi*frq_m*t + (phs_f - phs_c)/2.)) + ofs

t_a, x_a, t_b, x_b = np.loadtxt('./data/beat.txt', unpack = True)
t_min = 1.8; t_max = 80.
# Preliminary plot to visualize the sub-interval of data to analyze
if tex: plt.rc('text', usetex=True); plt.rc('font', family='serif')
fig, (ax1, ax2) = plt.subplots(2, 1)
grid(ax1, xlab='Time [s]', ylab='Pendulum A [ADC counts]')
grid(ax2, xlab='Time [s]', ylab='Pendulum B [ADC counts]')
ax1.plot(t_a, x_a, 'gray', alpha=0.6); ax2.plot(t_b, x_b, 'gray', alpha=0.6)

t_a, x_a, t_b, x_b = lab.mesrange(t_a, x_a, t_b, x_b, t_min, t_max)
ax1.plot(t_a, x_a, 'k'); ax2.plot(t_b, x_b, 'k')
dt = np.full(t_a.shape, 2e-5); dx = 2*np.ones_like(x_a)

# fit pendulums
init_a = [np.std(x_a), 1.44, 0.03, 0., 0., np.mean(x_a), 60.]
init_b = [np.std(x_b), 1.42, 0.02, 0., 0., np.mean(x_b), 60.]

for time, pos, init in zip([t_a, t_b], [x_a, x_b], [init_a, init_b]):
    pars, covm, deff = propfit(time, dt, pos, dx, model=beat, p0=init)
    perr, pcor = errcor(covm)
    prnpar(pars, perr, model=beat)
    chisq, ndof, resn, sigma = chitest(pos, deff, beat(time, *pars),
                                         ddof=len(pars), gauss=True, v=True)

    # graphs
    fig, (axf, axr) = pltfitres(time, dt, pos, deff, model=beat, pars=pars)
    axf.set_ylabel('Position %c [ADC counts]' %('A' if all(time == t_a) else 'B'))
    if tix: lab.tick(axf, xmaj=5, ymaj=50)
    legend = axf.legend(loc='best')

    axr.set_xlabel('Time [s]', x=0.94)
    if tix: lab.tick(axr, xmaj=5, ymaj=2, ymin=0.5)
plt.show()