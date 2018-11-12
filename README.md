# Fast multidimensional Walsh-Hadamard transforms

[![Build Status](https://travis-ci.org/stevengj/Hadamard.jl.svg?branch=master)](https://travis-ci.org/stevengj/Hadamard.jl)
[![Coverage Status](https://coveralls.io/repos/stevengj/Hadamard.jl/badge.svg?branch=master)](https://coveralls.io/r/stevengj/Hadamard.jl?branch=master)

[![Hadamard](http://pkg.julialang.org/badges/Hadamard_0.6.svg)](http://pkg.julialang.org/?pkg=Hadamard&ver=0.6)

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
  same normalizations, respectively.    The natural-order transforms also
  have in-place variants `fwht_natural!` and `ifwht_natural!`.

## Hadamard matrices

We also provide a a function `hadamard(n)` which returns a Hadamard
matrix of order `n`, similar to the Matlab function of the same name.
The known Hadamard matrices up to size 256 are currently supported
(via a lookup table), along with any size that factorizes into
products of these known sizes and/or powers of two.

The return value of `hadamard(n)` is a matrix of `Int8` values.  If
you are planning to do matrix computations with this matrix, you may
want to convert to `Float64` first via `float(hadamard(n))`.

For many sizes, the Hadamard matrix is not unique; the `hadamard`
function returns an arbitrary choice.  For power-of-two sizes, the
choice is equivalent to `ifwht_natural(eye(n), 1)`.

You can pretty-print a Hadamard matrix as a table of `+` and `-`
(characters indicating the signs of the entries) via `Hadamard.printsigns`, e.g. `Hadamard.printsigns(hadamard(28))` for the 28×28 Hadamard matrix.

## Author

This package was written by [Steven G. Johnson](http://math.mit.edu/~stevenj/).
