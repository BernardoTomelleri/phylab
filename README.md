# Phy lab
A Python module for basic data analysis and curve fitting for Physics laboratory students.

## Philosophy behind this
Because this was written by someone starting to learn about experimental Physics
and Python (for students in a similar situation) all the functions in this module
try to use only the basic data structures and functionalities found in Python
that are briefly discussed with students (functions, arrays & not much else).
This is why you won't find classes, methods or even dicts in this library
and why the code inside may be very, very inefficient/ugly.
On the upside though, all the definitions inside should be easily comprehensible
and modifiable by anyone just starting out with programming.

### How did this library come about?
While writing the python scripts for individual experiments (at UniPi's Physics Lab)
I noticed that a significant amount of code was common to almost all of them.
I started writing down these common functions in a python module everytime this
happened, so that I could import them later to save a lot of time and hassle.
That's basically the story behind it: a collection of useful ideas I had while
learning how to analyze experimental data with Python's SciPy.
Now that it's grown into a powerful enough collection of tools, I decided to
release it for any and all students that may find it helpful for dealing
with similar problems.

## Contents
- The main module containing all the functions [lab](/lab.py).
- A simple showcase of a couple of things this library allows you to do. [circfit](/circfit.py)
-- Finding the best-fitting circle and ellipse for simulated or real sampled data points.
-- A quick comparison between using `curve_fit`, an algebraic circle fit (weighted [Coope] method) and an algebraic ellipse fit.
-- Plotting the 3d chi-square landscape for two parameters of the circle or the ellipse.
- Folder containing data that can be used in the demo(s).

For further examples of where this module can come in handy feel free to check
out [FFT] and [Lock-in detector]. A small paper (in italiano) on fitting,
computing Fourier transforms and/or simulating the effect of a Lock-in detector
on real sampled signals or internally generated ones.

### Model functions
A few definitions of the model functions more commonly encountered in the first
years of Physics lab (e.g. [dampened oscillator](/lab.py#L28), [square wave](/lab.py#L36))
along with a few digital filters (e.g. [Butterworth](/lab.py#L69)).
Right at the beginning so you can immediately start adding the models you need.

### Curve-fitting routines
Weighted least-square fitting accounting for uncertainties on x and y axes
with linear error propagation [propagated fit](/lab.py#L244), relying on
[scipy.optimize.curve_fit].
Weighted orthogonal distance regression thanks to [ODRPACK].
Weighted algebraic fits, like [ellipse fit](/lab.py#L354) and others.

### Fourier transform utilities
Functions for computing real and complex signal FFTs and other
quantities necessary for the calculation ([FFT](/lab.py#L431), [sampling](/lab.py#L512)),
applying window functions, and displaying their output through [Matplotlib]
([plotfft](/lab.py#L468)).

### data plotting
Instead of having to write multiple calls to function in order to:
activate axis minor ticks, setting their size, orientation and spacing,
placing grids, setting errorbar sizes, etc.. (stuff that's present in
almost all experimental graphs)
You can set some sensible defaults for plots of fitted data, so you can
do all of the above in a faster and less error-prone way.
([grid](/lab.py#L206), [plot fit &  residuals](/lab.py#L405))

### formatted printing of fit results
Simple print formatters that allow you to quickly display the results
of a fit and associated uncertainties with the desired number of significant
digits. ([print correlations](/lab.py#L156), [print parameters](/lab.py#L163))

### importing data from files
Load a selected range of data from (.txt, .csv, .py, etc.) files as [Numpy]
arrays, loop over files in a directory with a few calls to function.
([measured range](/lab.py#L506), [file loop](/lab.py#L533))

## Development
Any and all suggestions are always appreciated, If you want to contribute
in any way don't hesitate to contact me. I'm always happy to learn something
new, so if you know how to improve any part of the code, find something
that needs fixing or even if you'd like to see something added going forwards,
feel free to let me know.

## Installation
This library was written entirely in Python 3.x, but because of its entry
level design should be effortless to readapt to Python 2.x.
Should be completely OS independent, just download the main module
and import it in your script.

## License
Phy lab is licensed under the GNU General Public License v3.0 or later.

[//]: # (These are reference links used in the body of this note and get
stripped out when the markdown processor does its job. There is no need
to format nicely because it shouldn't be seen.
See - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [FFT]: <https://github.com/BernardoTomelleri/FFT/blob/master/fft_plot.py>
   [Lock-in detector]: <https://github.com/BernardoTomelleri/FFT/blob/master/lockin.py>
   [coope]: <https://ir.canterbury.ac.nz/bitstream/handle/10092/11104/coope_report_no69_1992.pdf?sequence=1&isAllowed=y>
   [SciPy]: <https://www.scipy.org/>   
   [NumPy]: <https://numpy.org/>
   [Matplotlib]: <https://matplotlib.org/stable/index.html>
   [ODRPACK]: <https://docs.scipy.org/doc/external/odrpack_guide.pdf>
   [scipy.optimize.curve_fit]: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>