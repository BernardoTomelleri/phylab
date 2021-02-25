# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:28:17 2019

@author: berna
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Lettura del file di dati in ingresso (da modificare con il percorso al
# vostro file in ingresso).
t, A, B = np.loadtxt('./data/beat.txt', unpack = True, usecols= (0,1,3))
dt = 0.00002
# Selezione dell'intervallo di tempo per il fit:
t_min = 1.8
t_max = 80.

# Extrazione dei dati corrispondenti all'intervallo di tempo selezionato
t1 = t[t>t_min]; st = t1[t1<t_max];
A1 = A[t>t_min]; sA = A1[t1<t_max];
B1 = B[t>t_min]; sB = B1[t1<t_max];

# Per prima cosa guardiamo i grafici dei due pendoli
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(1)
plt.subplot(211)
plt.ylabel('Pendolo A [u.a.]')
plt.grid(color = 'gray')
plt.plot(t, A, 'r') 
plt.plot(st, sA, 'c')
plt.subplot(212)
plt.xlabel('Tempo [{s}]')
plt.ylabel('Pendolo B [u.a.]')
plt.grid(color = 'gray')
plt.plot(t, B, 'g')
plt.plot(st, sB, 'c')

# Funzione di fit: oscillatore armonico smorzato

#def f(x, A, omega, tau, p, c):
#    return A*np.exp(-x/tau)*np.cos(omega*x +p) + c
def f(t, A, tau, omega_p, omega_b, p1, p2, c):
    return np.exp(-t/tau)*2*A*(np.cos((omega_p *t) + (p1+p2)/2.) * np.cos((omega_b *t) + (p2-p1)/2.)) + c
# Inizializzazione dei parametri per il pendolo A
# Attenzione: partire da un set di parametri appropriati e' fondamentale 
# per la convergenza del fit 
initParamsA = np.array([350., 30., 4.5, 0.07, 1., 1., 0.])

# Fit del pendolo A nell'intervallo selezionato
popt_A, pcov_A = curve_fit(f, st, sA, initParamsA, absolute_sigma = True)
A_fit, tau_fit, omega_fit_pA, omega_fit_bA, p1_fit_A, p2_fit_A, c_fit_A = popt_A
dA_fit, dtau_fit, domega_fit_pA, domega_fit_bA, dp1_fit_A, dp2_fit_A, dc_fit_A = np.sqrt(pcov_A.diagonal())
print('Omega p = %f +- %f rad/s' % (omega_fit_pA, domega_fit_pA))
print('Omega b = %f +- %f rad/s' % (omega_fit_bA, domega_fit_bA))

da = np.full(len(sA), 2)
#Calcolo del Chi quadro
chisq = (((sA - f(st, A_fit, tau_fit, omega_fit_pA, omega_fit_bA, p1_fit_A, p2_fit_A, c_fit_A))/da)**2).sum()
#Numnero di gradi di libertà, numero dei dati - numero dei vincoli (parametri retta)
#Corrisponde ai ddof n-1 della deviazione standard sul campione
ndof = len(sA) - len(initParamsA)
#Chi quadro ridotto, ovvero diviso per il numero di gradi di libertà
chirid = chisq/ndof
sigma = (chisq - ndof)/np.sqrt(2*ndof)
res = sA - f(st, A_fit, tau_fit, omega_fit_pA, omega_fit_bA, p1_fit_A, p2_fit_A, c_fit_A)
resnorm = res/(da)

print('Chi quadro/ndof = %f/%d [%+.1f]' % (chisq, ndof, sigma))
print('Chi quadro ridotto:', chirid)
# Realizzazione e salvataggio del grafico A
fig1,(ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'wspace':0.05, 'hspace':0.05, 'height_ratios': [3, 1]})
ax1.set_ylabel('Posizione A [ADC counts]')
ax1.grid(color = 'gray', linestyle = '--', alpha=0.7)
ax1.errorbar(st, sA, da, 0.001, marker='o', ms=2., elinewidth = 1.5, capsize=1.5, ls='', label='data $[%+.1f \, \sigma]$' %sigma, zorder=5)
ax1.plot(st, f(st, A_fit, tau_fit, omega_fit_pA, omega_fit_bA, p1_fit_A, p2_fit_A, c_fit_A), label='fit $\chi^2 = %.1f/%d$' %(chisq, ndof), zorder=10)
ax1.xaxis.set_major_locator(plt.MultipleLocator(5.))
ax1.xaxis.set_minor_locator(plt.MultipleLocator(1.))
ax1.yaxis.set_major_locator(plt.MultipleLocator(50.))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(10.))
ax1.tick_params(direction='in', length=5, width=1., top=True, right=True)
ax1.tick_params(which='minor', direction='in', width=1., top=True, right=True)
legend = ax1.legend(loc='best')

ax2.axhline(0, c='r', zorder=10)
ax2.errorbar(st, resnorm, 1. , dt, 'ko', elinewidth = 0.4, capsize=0.4, markersize=0.5, linestyle='', zorder = 5)
ax2.grid(color ='gray', linestyle = '--', alpha=0.7)
ax2.set_xlabel('Tempo [s]', x=0.94)
ax2.set_ylabel('Residui [a.u.]')
ax2.xaxis.set_major_locator(plt.MultipleLocator(5.))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(1.))
ax2.yaxis.set_major_locator(plt.MultipleLocator(2.))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax2.tick_params(direction='in', length=5, width=1., top=True, right=True)
ax2.tick_params(which='minor', direction='in', width=1., top=True, right=True)

# il fit del pendolo B si esegue allo stesso modo....

initParamsB = np.array([350., 30., 4.5, 0.069, 0., 0., 0.])

popt_B, pcov_B = curve_fit(f, st, sB, initParamsB, absolute_sigma = True)
B_fit, tau_fit, omega_fit_pB, omega_fit_bB, p1_fit_B, p2_fit_B, c_fit_B = popt_B
dB_fit, dtau_fit, domega_fit_pB, domega_fit_bB, dp1_fit_B, dp2_fit_B, dc_fit_B = np.sqrt(pcov_B.diagonal())
print('Omega p = %f +- %f rad/s' % (omega_fit_pB, domega_fit_pB))
print('Omega b = %f +- %f rad/s' % (omega_fit_bB, domega_fit_bB))

db = np.full(len(sB), 2)
#Calcolo del Chi quadro
chisq = (((sB - f(st, B_fit, tau_fit, omega_fit_pB, omega_fit_bB, p1_fit_B, p2_fit_B, c_fit_B))/db)**2).sum()
#Numnero di gradi di libertà, numero dei dati - numero dei vincoli (parametri retta)
#Corrisponde ai ddof n-1 della deviazione standard sul campione
ndof = len(sB) - len(initParamsB)
#Chi quadro ridotto, ovvero diviso per il numero di gradi di libertà
chirid = chisq/ndof
res = sB - f(st, B_fit, tau_fit, omega_fit_pB, omega_fit_bB, p1_fit_B, p2_fit_B, c_fit_B)
resnorm = res/(da)
sigma = (chisq - ndof)/np.sqrt(2*ndof)
print('Chi quadro/ndof = %f/%d \; [%+.1f]' % (chisq, ndof, sigma))
print('Chi quadro ridotto:', chirid)
# Realizzazione e salvataggiodd del grafico B
fig2,(ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'wspace':0.05, 'hspace':0.05, 'height_ratios': [3, 1]})
ax1.set_ylabel('Posizione B [ADC counts]')
ax1.grid(color = 'gray', linestyle = '--', alpha=0.7)
ax1.errorbar(st, sB, db, 0.001, marker='o', ms=2., elinewidth = 1.5, capsize=1.5, ls='', label='data $[%+.1f \, \sigma]$' %sigma, zorder=0)
ax1.plot(st, f(st, B_fit, tau_fit, omega_fit_pB, omega_fit_bB, p1_fit_B, p2_fit_B, c_fit_B), label='fit $\chi^2 = %.1f/%d$' %(chisq, ndof), zorder=10)
ax1.xaxis.set_major_locator(plt.MultipleLocator(5.))
ax1.xaxis.set_minor_locator(plt.MultipleLocator(1.))
ax1.yaxis.set_major_locator(plt.MultipleLocator(50.))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(10.))
ax1.tick_params(direction='in', length=5, width=1., top=True, right=True)
ax1.tick_params(which='minor', direction='in', width=1., top=True, right=True)
legend = ax1.legend(loc='best')

ax2.axhline(0, c='r', zorder=10)
ax2.errorbar(st, resnorm, 1. , dt, 'ko', elinewidth = 0.4, capsize=0.4, markersize=0.5, linestyle='', zorder = 5)
ax2.grid(color ='gray', linestyle = '--', alpha=0.7)
ax2.set_xlabel('Tempo [s]', x=0.94)
ax2.set_ylabel('Residui [a.u.]')
ax2.xaxis.set_major_locator(plt.MultipleLocator(5.))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(1.))
ax2.yaxis.set_major_locator(plt.MultipleLocator(2.))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax2.tick_params(direction='in', length=5, width=1., top=True, right=True)
ax2.tick_params(which='minor', direction='in', width=1., top=True, right=True)
plt.show()