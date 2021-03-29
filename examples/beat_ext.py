# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:43:34 2021

@author: berni
"""
from phylab import (np, plt, grid, propfit, errcor, prnpar, chitest, pltfitres)
import phylab as lab

''' Variables that control the script '''
tix = False # manually choose spacing between axis ticks
tex = False # LaTeX typesetting maths and descriptions
out = False # re-fit the data after removing outliers
fft = True # Compute and display fft of data array
chi = True # Plot the chi square landscape for two parameters

def beat(t, A, frq_d, frq_m, phs_f=0, phs_c=0, ofs=0, tau=1):
    return np.exp(-t/tau)*2*A*(np.cos(np.pi*frq_d*t + (phs_f + phs_c)/2.)
                             * np.cos(np.pi*frq_m*t + (phs_f - phs_c)/2.)) + ofs

t_a, x_a, t_b, x_b = np.loadtxt('./data/beat.txt', unpack = True)
t_min = 1.8; t_max = 80.

# Preliminary plot to visualize the sub-interval of data to analyze
lab.rc.typeset(usetex=tex, fontsize=12)
fig, (ax1, ax2) = plt.subplots(2, 1)
grid(ax1, xlab='Time [s]', ylab='Pendulum A [ADC counts]')
grid(ax2, xlab='Time [s]', ylab='Pendulum B [ADC counts]')
ax1.plot(t_a, x_a, 'gray', alpha=0.6); ax2.plot(t_b, x_b, 'gray', alpha=0.6)

t_a, x_a, t_b, x_b = lab.mesrange(t_a, x_a, t_b, x_b, t_min, t_max)
ax1.plot(t_a, x_a, 'k'); ax2.plot(t_b, x_b, 'k')

# Prep before fitting pendulums
dt = np.full(t_a.shape, 2e-5); dx = 2*np.ones_like(x_a)
init_a = [np.std(x_a), 1.44, 0.03, 0., 0., np.mean(x_a), 60.]
init_b = [np.std(x_b), 1.42, 0.02, 0., 0., np.mean(x_b), 60.]

for time, pos, init in zip([t_a, t_b], [x_a, x_b], [init_a, init_b]):
    pars, covm, deff = propfit(beat, time, pos, dt, dx, p0=init)
    perr, pcor = errcor(covm)
    prnpar(pars, perr, model=beat)
    chisq, ndof, resn, sigma = chitest(beat(time, *pars), pos, unc=deff,
                                         ddof=len(pars), gauss=True, v=True)

    # Standard graph with residuals
    fig, (axf, axr) = pltfitres(beat, time, pos, dt, deff, pars=pars)
    axf.set_ylabel('Position %c [ADC counts]' %('A' if all(time == t_a) else 'B'))
    if tix: lab.tick(axf, xmaj=5, ymaj=50)
    legend = axf.legend(loc='best')

    axr.set_xlabel('Time [s]', x=0.94)
    if tix: lab.tick(axr, xmaj=10, ymaj=2, ymin=0.5)

    if out:
        timein, posin, dtin, dpin, mask = lab.outlier(beat, time, pos, dt,
                                                      deff, pars, thr=4, out=True)
        # propagated fit again without considering outliers
        pars, covm, dpin = propfit(beat, timein, posin, dtin, dpin, p0=pars)
        perr, pcor = errcor(covm)
        prnpar(pars, perr, model=beat)
        chisq, ndof, resn, sigma = chitest(beat(timein, *pars), posin, unc=dpin,
                                           ddof=len(pars), gauss=True, v=True)

        # Graph with outliers
        figout, (axf, axr) = pltfitres(beat, time, pos, dx=dt, dy=deff, pars=pars, in_out=mask)
        axf.set_ylabel('Position %c [ADC counts]' %('A' if all(time == t_a) else 'B'))
        if tix: lab.tick(axf, xmaj=10, ymaj=50)
        legend = axf.legend(loc='best')

        axr.set_xlabel('Time [s]', x=0.94)
        if tix: lab.tick(axr, xmaj=5, ymaj=2, ymin=0.5)

    if fft:
        freq, tran, fres, frstd = lab.FFT(time, signal=(pos - pars[-2]), window=np.kaiser, beta=None)
        figfrq, (ax1, ax2) = lab.plotfft(freq, tran, norm=True, mod_ph=True, dB=True)
        if tix:
            ax1.set_xlim(0, 5)
            lab.tick(ax1, xmaj=0.5, ymaj=20)

    if chi:
        a_range = np.linspace(pars[1] - 3e3*perr[1], pars[1] + 3e3*perr[1], 100)
        b_range = np.linspace(pars[2] - 3e3*perr[2], pars[2] + 3e3*perr[2], 100)
        chi_ab = lab.chisq(model=beat, x=time, y=pos, alpha=a_range, beta=b_range,
                           varnames = ['frq_d', 'frq_m'], pars=pars, dy=dx)

        fig3d, ax3d = lab.plot3d(x=a_range, y=b_range, z=chi_ab, xlab='$f_d$ [Hz]',
                                 ylab='$f_m$ [Hz]', zlab='$\chi^2(f_d, f_m)$')
        ax3d.set_zlim(0, None)
        if tex: ax3d.set_title(r'$\displaystyle \chi^2(f_d, f_m) = \sum_{i=1}^{n = %d}'  %len(x_a) +
                               r'\left( \frac{x_i - 2A\left\{ \cos\left[ \pi f_d t_i +'
                               r'\frac{ (\phi_1 + \phi_2)}{2}\right] \cdot \cos\left['
                               r'\pi f_m t_i + \frac{(\phi_1 - \phi_2)}{2}\right] \right\} }'
                               r'{\sigma_{x_i}} \right)^2$', fontsize=12)
plt.show()
