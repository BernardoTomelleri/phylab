![phylab_logo](/svgs/logo.svg)

Phy lab
=======
A Python package for basic data analysis and curve fitting for Physics laboratory students.

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
- The main module containing all the functions. [lab](/phylab/lab.py)
- A simple showcase of a couple of things this library allows you to do. [circfit](/phylab/circfit.py)
  - Finding the best-fitting circle and ellipse for simulated or real sampled data points.
  - A quick comparison between using `curve_fit`, an algebraic circle fit
   (weighted [Coope] method) and an algebraic ellipse fit.
  - Plotting the <img src="svgs/0fb48cfe3fc8c14f5b77eba3ba39a718.svg?invert_in_darkmode" align=middle width=53.495080649999984pt height=26.76175259999998pt/> surface for a pair of parameters <img src="svgs/5f8c6707c3c404791835c4d82736cf4f.svg?invert_in_darkmode" align=middle width=23.04983339999999pt height=22.831056599999986pt/> of the circle or the ellipse.
- Folder containing further examples and data that can be used in the demos. [examples](/examples)

For another example of where this package can come in handy feel free to check
out [FFT] and [Lock-in detector]. A small paper (in italiano) on fitting,
computing Fourier transforms and/or simulating the effect of a Lock-in detector
on real sampled signals or internally generated ones.

### model functions
A few definitions of the model functions more commonly encountered in the first
years of Physics lab (e.g. [dampened oscillator], [square wave])
along with a few digital filters (e.g. [Butterworth]).
Right at the beginning of the module so you can immediately start adding
the models you need.

### testing and printing of fit results
Goodness of fit tests, evaluation of parameter uncertainties and
correlations  ([chi-square test], [errcor]).
Simple print formatters that allow you to quickly display the results
of a fit and associated uncertainties with the desired number of significant
digits. ([print correlations], [print parameters])

### curve-fitting routines
Weighted least-square fitting accounting for uncertainties on x and y axes
with linear error propagation ([propagated fit]), relying on [scipy.optimize.curve_fit].
Weighted orthogonal distance regression thanks to [ODRPACK].
Weighted algebraic fits (like [ellipse fit] and others).

### Fourier transform utilities
Functions for computing FFTs of real and complex signals and other
associated quantities ([FFT], [sampling], [FWHM]), applying window functions
and displaying their output through [Matplotlib] ([plotfft]).

### data plotting
Instead of having to write multiple calls to function in order to:
activate axis minor ticks, setting their size, orientation and spacing,
placing grids, setting errorbar sizes, etc.. (stuff that's present in
almost all experimental graphs)
You can set some sensible defaults for plots of fitted data, so you can
do all of the above in a faster and less error-prone way.
([grid], [plot fit &  residuals])

### importing data from files
Load a selected range of data from (.txt, .csv, .py, etc.) files as [Numpy]
arrays, loop over files in a directory with a few calls to function
([measured range], [file loop]).

## Installation
This library was written entirely in Python 3.x, but because of its entry
level design should be effortless to readapt to Python 2.x.
Should be completely OS independent.
The cleanest/easiest way to install and manage phylab is using `pip`:
```
pip install --index-url https://test.pypi.org/simple/ --no-deps phylab
```
Then simply import the main module with
```
from phylab import lab
```

Alternatively you can clone the repository and import the main module directly from
within your script by calling
```
import phylab as lab
```

## Using Phy lab
For a quick guide on how to use this library and to show just how much of a difference
using these few functions can have, compare [beat](/examples/beat.py) and
[beat_naive](/examples/beat_naive.py).
These two scripts do the same thing, but the first one is three times shorter
at 50 lines, runs ~ 0.2 seconds (30%) faster using less memory and can be
easily extended to work with more than 2 datasets, remove outliers,
compute FFT and so on... As you can see for example in [beat_ext](/phylab/beat_ext.py).

## Development
Any and all suggestions are always appreciated, If you want to contribute
in any way don't hesitate to contact me. I'm always happy to learn something
new, so if you know how to improve any part of the code, find something
that needs fixing or even if you'd like to see something added going forwards,
feel free to let me know (here or at bernardo.tomelleri@gmail.com).

## License
Phy lab is licensed under the GNU General Public License v3.0 or later.

[//]: # (These are reference links used in the body of the readme and get
stripped out by the markdown processor.
See - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [FFT]: <https://github.com/BernardoTomelleri/FFT/blob/master/fft_plot.py>
   [Lock-in detector]: <https://github.com/BernardoTomelleri/FFT/blob/master/lockin.py>
   [coope]: <https://ir.canterbury.ac.nz/bitstream/handle/10092/11104/coope_report_no69_1992.pdf?sequence=1&isAllowed=y>
   [dampened oscillator]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L28>
   [square wave]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L36>
   [Butterworth]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L69>
   [chi-square test]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L99>
   [errcor]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L147>
   [print correlations]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L156>
   [print parameters]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L163>
   [propagated fit]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L244>
   [ellipse fit]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L380>
   [FFT]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L431>
   [sampling]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L512>
   [FWHM]: <https://github.com/BernardoTomelleri/phylab/blob/b03e131d2007a1ebe2d100dcd2d2d0f3de764fe3/phylab.py#L201>
   [plotfft]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L468>
   [grid]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L206>
   [plot fit &  residuals]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L405>
   [measured range]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L506>
   [file loop]: <https://github.com/BernardoTomelleri/phylab/blob/66c6b772e1d3ea614c796b8c146bf99b1f1540c5/lab.py#L533>
   [SciPy]: <https://www.scipy.org/>   
   [NumPy]: <https://numpy.org/>
   [Matplotlib]: <https://matplotlib.org/stable/index.html>
   [ODRPACK]: <https://docs.scipy.org/doc/external/odrpack_guide.pdf>
   [scipy.optimize.curve_fit]: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>