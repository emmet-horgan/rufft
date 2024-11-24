# Rufft

[![ci](https://github.com/emmet-horgan/rufft/workflows/ci/badge.svg)](https://github.com/emmet-horgan/rufft/actions?query=workflow%3Aci)
[![](https://img.shields.io/crates/v/rufft.svg)](https://crates.io/crates/rufft)
[![](https://img.shields.io/crates/l/rufft.svg)](https://crates.io/crates/rufft)
[![](https://docs.rs/rufft/badge.svg)](https://docs.rs/ruﬁfft/)

Rufft is a purely rust implementation of several common fast fourier transform algorithms. The libary functions operate on collection types which implement a library trait called `Iterable` which provides a method to get an iterator and to get the length of the collection. In the future other convenience modules will be added for things like waveform generation and improvements to the fft implementation to support things like paralleization, and SIMD acceleration

## Usage

```rust
use rufft::{traits::Fft, Complex};

// Get the FFT of a Vec
let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let farr: Vec<Complex<f64>> = arr.fft();
```

## Motivation
The project originally started for my own educational purposes to both improve my rust skills and also to implement fast fourier transform algorithms. As time went on I looked at some other more mature rust based FFT libaries such as,
* [RustFFT](https://crates.io/crates/rustfft)
* [tfhe-fft](https://crates.io/crates/tfhe-fft)

These projects are great and very performent focussed especially `rustfft` but I realized that I wanted to use something that reflected a much simpler API like `scipy` in the python ecosystem but was still relatively performant, something that could be used in `no_std` environments, static dispatch based rather than dynamic dispatch and is compatible with most collection types and gives control over what types are used internally. 
