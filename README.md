# Fast multidimensional Walsh-Hadamard transforms

This package provides functions to compute fast Walsh-Hadamard transforms
in Julia, for arbitrary dimensions and arbitrary power-of-two transform sizes,
with the three standard orderings: natural (Hadamard), dyadic (Paley), and
sequency (Walsh) ordering.

It works by calling Julia's interface to the [FFTW](http://www.fftw.org/)
library, and can often be orders of magnitude faster than the corresponding
`fwht` functions in the Matlab signal-processing toolbox.

## Installation

Within Julia, just use the package manager to run `Pkg.add("Hadamard")` to
install the files.

## Usage

After installation, the `using Hadamard` statement will import the names
in the Hadamard module so that you can call the function below.

* The function `fwht(X)` computes the fast Walsh-Hadamard transform
  (WHT) of the multidimensional array `X` (of real or complex numbers),
  returning its output in sequency order.  The inverse transform is
  `ifwht(X)`.

By default, `fwht` and `ifwht` compute the *multidimensional* WHT, which
consists of the ordinary (one-dimensional) WHT performed along each dimension
of the input.  To perform only the 1d WHT along dimension `d`, you can
can instead use `fwht(X, d)` and `ifwht(X, d)` functions.  More generally,
`d` can be a tuple or array or dimensions to transform.

The sizes of the transformed dimensions *must* be powers of two, or an
exception is thrown.  The non-transformed dimensions are arbitrary.  For
example, given a 16x20 array `X`, `fwht(X,1)` is allowed but `fwht(X,2)` is
not.

These functions compute the WHT normalized similarly to the `fwht` and
`ifwht` functions in Matlab.  Given the Walsh functions, which have values
of +1 or -1, `fwht` multiplies its input by the Walsh functions and divides
by `n` (the length of the input) to obtain the coefficients of each Walsh
function in the input.  `ifwht` multiplies its inputs by the Walsh functions
and sums them to recover the signal, with no `n` factor.

* Instead of sequency order, one can also compute the WHT in the natural
  (Hadamard) ordering with `fwht_natural` and `ifwht_natural`, or in the
  dyadic (Paley) ordering with `fwht_dyadic` and `ifwht_dyadic`.  These
  functions take the same arguments as `fwht` and `ifwht` and have the
  same normalizations, respectively.

* Also provided is a function `hadamard(n)` which returns the
  (natural-order) Hadamard matrix of order `n`, similar to the Matlab
  function of the same name.  Currently, `n` must be a power of two,
  in which case this function is equivalent to `ifwht_natural(eye(n), 1)`.

## Author

This package was written by [Steven G. Johnson](http://math.mit.edu/~stevenj/).
