# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:48:23 2021

@author: berni
"""
from matplotlib import cm
from lab import (np, plt, chitest, grid, errcor, std_unc, prnpar, prncor,
                 chisq, pltfitres, tick, coope, circ, crcfit, elpfit)

''' Variables that control the script '''
gen = True # generate measured points around circumference
con = False # show confidence interval for circle fit
tex = True # LaTeX typesetting maths and descriptions
# (center coordinates x, y, radius or major semiaxis, minor semiaxis, tilt /x)
init = (6, 6, 10, None, None, 1); npts = 50

def parel(Xc, Yc, a, b=None, tilt=None, arc=1, step=1000):
    if not b: b = a
    elif b > a: a, b = b, a
    theta = np.linspace(0, 2*np.pi*arc, step)
    if tilt:
        x = np.cos(tilt)*a*np.cos(theta) - np.sin(tilt)*b*np.sin(theta) + Xc
        y = np.sin(tilt)*a*np.cos(theta) + np.cos(tilt)*b*np.sin(theta) + Yc
    else:
        x = a*np.cos(theta) + Xc
        y = b*np.sin(theta) + Yc
    return x, y

def canel(impars):
    A, B, C, D, E = impars; F = -1
    DEL = B**2 - 4*A*C; DIS = 2*(A*E**2 + C*D**2 - B*D*E + DEL*F)
    a = -np.sqrt(DIS*((A+C) + np.sqrt((A - C)**2 + B**2)))/DEL
    b = -np.sqrt(DIS*((A+C) - np.sqrt((A - C)**2 + B**2)))/DEL
    Xc = (2*C*D - B*E)/DEL; Yc = (2*A*E - B*D)/DEL
    tilt = np.arctan(1./B *(C - A - np.sqrt((A - C)**2 + B**2))) if B != 0 else (0 if B == 0 and A < C else 0.5*np.pi)
    return np.asarray([Xc, Yc, a, b, tilt])

def circle(coords, Xc=0, Yc=0, R=1):
    x, y = coords
    return 2*Xc*x + 2*Yc*y + (R**2 - Xc**2 - Yc**2)

if gen:
    data = np.asarray(parel(*init, step=npts))
    noise = np.random.normal(loc=0, scale=0.1*init[2], size=data.shape)
    data += noise
else: data = np.loadtxt('./data/circ.txt', unpack=True)
dx = std_unc(data[0]); dy = std_unc(data[1]);
rsq = (data**2).sum(axis=0); dr = np.sqrt(dx**2 + dy**2)

# weighted Coope method
cen, rad = coope(data, weights=1./dr)

# Scipy's curve_fit
pars, covm = crcfit(data, p0=init[:3])
perr, pcor = errcor(covm)
prnpar(pars, perr, circ)

# algebraic elliptical fit
sol, chi2, pcov = elpfit(data, uncerts=dr)
p = canel(sol).squeeze()
if chi2.size > 0:
    unc, cor = errcor(pcov)
    prnpar(p, unc, manual=['Xc', 'Yc', 'a', 'b', 'tilt'])
else: unc = np.zeros_like(sol)

if tex: plt.rc('text', usetex=True); plt.rc('font', family='serif')
xCop, yCop = parel(*cen, rad); xfit, yfit = parel(*pars); xEl, yEl = parel(*p)
fig, axs = plt.subplots(1,2, sharex=True); axc = axs[0]; axe = axs[1]
for ax in axs:
    grid(ax); ax.axis('equal')
    ax.errorbar(*data, dy, dx, 'ko', ms=1.2, elinewidth=0.8, capsize= 1.1,
                ls='',label='data', zorder=5)

# Plot fitted circles
axc.plot(xCop, yCop, c='c', ls='--', lw=0.8, zorder=10, alpha=0.6, label='Coope radius')
axc.plot(cen[0], cen[1], c='c', marker='.', label='Coope center')
axc.plot(xfit, yfit, c='y', ls='--', lw=0.8, zorder=10, alpha=0.6, label='circle fit radius')
axc.errorbar(*pars[:-1], *perr[:-1],'yo', ms=1.5, elinewidth=0.9, capsize= 1.2,
        ls='', label='circle fit center')
if con: 
    axc.plot(*parel(*pars[:-1], a=pars[-1] + perr[-1]), c='r', ls='--', lw=0.8,
            zorder=10, alpha=0.6, label='circle fit $r + dr$')
    axc.plot(*parel(*pars[:-1], a=pars[-1] - perr[-1]), c='b', ls='--', lw=0.8,
            zorder=10, alpha=0.6, label='circle fit $r - dr$')

# Plot fitted and original (noiseless) ellipses
axe.plot(xEl, yEl, c='m', ls='--', lw=0.8, zorder=10, alpha=0.6, label='best-fit ellipse')
axe.errorbar(p[0], p[1], unc[1], unc[0], 'mo', ms=1.5, elinewidth=0.9,
            capsize= 1.2, label='ellipse center')
if gen: 
    axe.plot(*parel(*init, step=npts), c='pink', ls='-', lw=1,
                zorder=10, alpha=0.6, label='generated ellipse')
    axe.plot(init[0], init[1], c='pink', marker='.', label='generated center')
if con:
     axe.plot(*parel(*p[:2], a=p[2] + unc[2], b=p[3] + unc[3], tilt=p[-1]), ls='--', lw=0.8,
            zorder=10, alpha=0.6, label='ellipse fit $a + da$, $b + db$')
     axe.plot(*parel(*p[:2], a=p[2] - unc[2], b=p[3] - unc[3], tilt=p[-1]), ls='--', lw=0.8,
             zorder=10, alpha=0.6, label='ellipse fit $a - da$, $b - db$')
for ax in axs: ax.legend(loc ='best')

# Data for the 3D plot
xcen = np.linspace(pars[0] - 10*perr[0], pars[0] + 10*perr[0], 50)
ycen = np.linspace(pars[1] - 10*perr[1], pars[1] + 10*perr[1], 50)
Z = np.array(chisq(x=data, y=rsq, model=circle, alpha=xcen, beta=ycen,
                   varnames = ['Xc', 'Yc'], pars=pars, dy=None))
X, Y = np.meshgrid(xcen, ycen)

# Plot the Chi square surface for circle fit
fig = plt.figure()
frame = fig.add_subplot(projection='3d')
frame.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                   linewidth=0, antialiased=False, alpha=0.6)
cset = frame.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.jet); frame.set_zlim(0, np.max(Z))
frame.set_xlabel('$x_c$'); frame.set_ylabel('$y_c$'); frame.set_zlabel('$\chi^2(x_c, y_c)$')
if tex: frame.set_title(r'$\displaystyle \chi^2(x_c, y_c) = \sum_{i=1}^n'
                r'\left(\frac{r_i^2 - (2x_c x_i + 2y_c y_i + R_c^2)}{\sigma_{r^2_i}}\right)^2$'
                r'of circle fit $(x_i^2 + y_i^2 := r_i^2)$' , fontsize=12)
plt.show()