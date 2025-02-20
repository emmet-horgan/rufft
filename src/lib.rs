#![cfg_attr(not(feature = "std"), no_std)]
//! Rufft is a pure rust signal processing library which implements fast fourier transform
//! algorithms. The primary aims of the library are to be compatible with most collection
//! types, be generic over the floating point type in use and be usable in `no_std` environments. 
//!
//!
//!
//! ### Usage
//! Rufft provides the trait `rufft::traits::Fft` which is blanket implemented on types which implement
//! the `rufft::traits::Iterable` trait over floating point types i.e. implement the `num_traits::Float` type.
//! The `Fft` trait provides an `fft` method which returns a collection of the fft results. The compiler
//! requires some type information to determine the output type e.g. `Vec<f64>` is not the same type
//! as `Vec<Complex<f64>>`. The output type must also implement the `Iterable` trait but over 
//! `Complex<F: num_traits::Float>` values. 
//!
//! ```
//! // Perform an fft on a Vec of floats
//! use rufft::{Complex, traits::Fft};
//!
//! let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let _: Vec<Complex<f64>> = arr.fft();
//! ```
//!
//! Rufft also exposes individual FFT algorithms in the `fft` module. Currently at the time
//! of writing only the basic discrete fourier transfrom, `dft`, the cooley-tukey fft 
//! algorithm `fft::ct::fft` and the chirp-z fft `fft::czt::fft`. The inverse transform is 
//! currently unsupported for the Chirp-Z transform. I am still learning about fast fourier
//! transform algorithms and will add more as time goes on. Any contributions there would be
//! appreciated.
//! 
//!
//! ```
//! // Computes the fft using the chirp-z transform
//! use rufft::{Complex, fft::czt};
//! 
//! let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let _: Vec<Complex<f64>> = czt::fft(&arr);
//! ```
//!
//! ### Feature Flags
//!
//! * `std` (Enabled by default)
//!
//!     Links with rust's std crate, enables `std` feature in dependecies and 
//!     provides a `Iterable` implementation for Vec. Vec technically is from the 
//!     `alloc` crate and re-exported in `std` but this will do for now
//!
//!     
//! * `ndarray`
//!
//!     Re-exports the ndarray scientific computing crate and provides an 
//!     `Iterable` trait implementation for the `ndarray::Array1` type
//!



pub mod fft;
pub mod traits;
pub mod itertools;
pub mod waveforms;

pub use num_complex::Complex;
pub use num_traits::{ Float, FloatConst };
pub use num_complex;
pub use num_traits;

#[cfg(all(feature = "std", feature = "ndarray"))]
pub use ndarray;

#[cfg(test)]
mod test_utils;
