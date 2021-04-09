# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:48:23 2021

@author: berni
"""
from phylab import (np, plt, grid, std_unc, prnpar, errcor, Ell_coords)
import phylab as lab

''' Variables that control the script '''
gen = False # generate measured points around circumference
con = False # show confidence interval for circle fit
tex = False # LaTeX typesetting maths and descriptions
# (center coordinates x, y, radius or major semiaxis, minor semiaxis, tilt /x)
init = (1, 2, 3, None, None, 1); npts = 50

def circ_dist(coords, Xc=0, Yc=0, R=1):
    x, y = coords
    return 2*Xc*x + 2*Yc*y + (R**2 - Xc**2 - Yc**2)

def elps_dist(coords, Xc=0, Yc=0, a=1, b=1, angle=0):
    x, y = coords
    A, B, C, D, E, F = lab.Ell_std2imp(Xc, Yc, a, b, angle)
    return A*x**2 + B*x*y + C*y**2 + D*x + E*y + F

def test_circle_fit():
    """
    This test was extracted from the paper: "Least-Squares Circle" Fit
    by Randy Bullock. It test the correctness of the fit.
    
    """
    x = np.array([0.000, 0.500, 1.000, 1.500, 2.000, 2.500, 3.000])
    y = np.array([0.000, 0.250, 1.000, 2.250, 4.000, 6.250, 9.000])
    coords = np.array([x, y])
    (xc, yc), r = lab.coope(coords)
    assert xc + 11.839 < 1e-3 and yc - 8.4464 < 1e-4 and r - 14.686 < 1e-3

def test_ellipse_fit():
    """
    This test was extracted from the paper: "Least-Squares Fitting of Circles
    and Ellipses" by W. Gander et al. It test the correctness of the fit.
    
    """
    x = np.array([1, 2, 5, 7, 9, 3, 6, 8])
    y = np.array([7, 6, 8, 7, 5, 7, 2, 4])
    coords = np.array([x, y])
    sol, chisq, pcov = lab.elpfit(coords)
    xc, yc, a, b, angle = sol
    assert xc - 2.6996 < 1e-4 and yc - 3.8160 < 1e-4 and a - 6.5187 < 1e-4 and b - 3.0319 < 1e-4
    
if gen:
    data = np.asarray(Ell_coords(*init, step=npts))
    noise = np.random.normal(loc=0, scale=0.1*init[2], size=data.shape)
    data += noise
else: data = np.loadtxt('./data/circ.txt', unpack=True)
dx = std_unc(data[0]); dy = std_unc(data[1]);
rsq = (data**2).sum(axis=0); dr = np.sqrt(dx**2 + dy**2)

# weighted Coope method
cen, rad = lab.coope(data, weights=1./dr)

# Scipy's curve_fit
pars, covm = lab.crcfit(data, p0=init[:3])
perr, pcor = errcor(covm)
prnpar(pars, perr, circ_dist)

# algebraic elliptical fit
sol, chi2, pcov = lab.elpfit(data, uncerts=dr)
p = lab.Ell_imp2std(*np.append(sol, -1)).squeeze()
if chi2.size > 0:
    unc, cor = lab.errcor(pcov)
    prnpar(p, unc, manual=['Xc', 'Yc', 'a', 'b', 'angle'])
else: unc = np.zeros_like(sol)

lab.rc.typeset(usetex=tex, fontsize=12)
xCop, yCop = Ell_coords(*cen, rad); xfit, yfit = Ell_coords(*pars); xEl, yEl = Ell_coords(*p)
fig, axs = plt.subplots(1, 2, sharex=True); axc = axs[0]; axe = axs[1]
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
    axc.plot(*Ell_coords(*pars[:-1], a=pars[-1] + perr[-1]), c='r', ls='--', lw=0.8,
            zorder=10, alpha=0.6, label='circle fit $r + dr$')
    axc.plot(*Ell_coords(*pars[:-1], a=pars[-1] - perr[-1]), c='b', ls='--', lw=0.8,
            zorder=10, alpha=0.6, label='circle fit $r - dr$')

# Plot fitted and initially generated (noiseless) ellipses
axe.plot(xEl, yEl, c='m', ls='--', lw=0.8, zorder=10, alpha=0.6, label='best-fit ellipse')
axe.errorbar(p[0], p[1], unc[1], unc[0], 'mo', ms=1.5, elinewidth=0.9,
            capsize= 1.2, label='ellipse center')
if gen:
    axe.plot(*Ell_coords(*init, step=npts), c='pink', ls='-', lw=1,
                zorder=10, alpha=0.6, label='generated ellipse')
    axe.plot(init[0], init[1], c='pink', marker='.', label='generated center')
if con:
     axe.plot(*Ell_coords(*p[:2], a=p[2] + unc[2], b=p[3] + unc[3], angle=p[-1]), ls='--', lw=0.8,
            zorder=10, alpha=0.6, label='ellipse fit $a + da$, $b + db$')
     axe.plot(*Ell_coords(*p[:2], a=p[2] - unc[2], b=p[3] - unc[3], angle=p[-1]), ls='--', lw=0.8,
             zorder=10, alpha=0.6, label='ellipse fit $a - da$, $b - db$')
for ax in axs: ax.legend(loc ='best')

# Plot the Chi square surface for circle's center coordinates
a_range = np.linspace(pars[0] - 10*perr[0], pars[0] + 10*perr[0], 50)
b_range = np.linspace(pars[1] - 10*perr[1], pars[1] + 10*perr[1], 50)
chi_ab = lab.chisq(model=circ_dist, x=data, y=rsq, alpha=a_range, beta=b_range,
                   varnames = ['Xc', 'Yc'], pars=pars, dy=None)

fig3d, ax3d = lab.plot3d(x=a_range, y=b_range, z=chi_ab, xlab='$x_c$', ylab='$y_c$',
                         zlab='$\chi^2(x_c, y_c)$')
ax3d.set_zlim(0, None)
if tex: ax3d.set_title(r'$\displaystyle \chi^2(x_c, y_c) = \sum_{i=1}^n'
                r'\left(\frac{r_i^2 - (2x_c x_i + 2y_c y_i + R_c^2)}{\sigma_{r^2_i}}\right)^2$'
                r'of circle fit $(x_i^2 + y_i^2 := r_i^2)$' , fontsize=12)
plt.show()
