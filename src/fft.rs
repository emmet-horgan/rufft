//! The `fft` module itself contains some basic functions ;ole `dft`, and `idft` functions. Other fast 
//! fourier transform algorithms are exported through crates like,
//!     
//! | Algorithm | Module Name |
//! | --------- | ----------- |
//! | Cooley-Tukey | `ct` | 
//! | Chirp-Z Transform (Bluestein's Algorithm) | `czt` | 
//! 
//! The most common use case tends to be computing the FFT of a real-valued input collection
//! producting a complex output collection. The opposite for computing the IFFT. Thus,
//! by default implementations should expose an `fft`, and `ifft` function meeting the 
//! most common use case, and a purely complex implementation is exposed through the 
//! algorithm's `complex` module. For example, the cooley-tukey implementation exposes
//! * `ct::fft`
//! * `ct::ifft`
//! * `ct::complex::fft`
//! * `ct::complex::ifft`
//! 
pub mod ct;
pub mod czt;
pub mod complex;
use num_integer::Integer;
use num_complex::Complex;
use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive, NumAssignOps };
use core::ops::IndexMut;
use crate::traits::{ ComplexFloatIterable, FloatIterable };

/// Compute the discrete time fourier transform of the real valued input collection.
/// Returns a closure which accepts a collection of sample frequencies and returns 
/// a collection of the fft values
pub fn dtft<F, I, C>(x: I) -> impl Fn(I) -> C
where 
    F: Float + FloatConst + NumAssign + 'static,
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    usize: AsPrimitive<F>,
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
{
    move |samples: I| -> C {
        samples.into_iter().map(|&w|{
            x.into_iter().enumerate().map(|(i, &f)| {
                let phase = Complex::<F>::new(F::zero(), -i.as_() * w);
                Complex::<F>::new(f, F::zero()) * phase.exp()
            }).sum()
        }).collect()
    }
}

/// Computes the inverse discrete fourier transform of the real valued input 
/// collection.
/// The output *is* normalized 
pub fn idft<F, I, C>(x: &I) -> C
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F>, C: FloatIterable<F>,
    usize: AsPrimitive<F>,
{
    complex::idft_internal(x).iter().map(|x| x.re).collect()
}

/// Computes the discrete fourier tranform on the real valued input collection
pub fn dft<F, I, C>(x: &I) -> C
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterable<F>, C: ComplexFloatIterable<F>,
    usize: AsPrimitive<F>,
{
    let n = x.len();
    let zero = F::zero();
    let twopi = F::TAU();
    x.iter().enumerate().map(|(i, _)|{ // Change to a range of some kind
        x.iter().enumerate().map(|(j, f)| {
            let phase = Complex::<F>::new(zero, -(twopi * j.as_() * i.as_()) / n.as_());
            Complex::<F>::new(*f, zero) * phase.exp()
        }).sum()
    }).collect()
}

/// Computes the frequency values associated fft based on `n` the length
/// of the collection and `d` the sampling period
pub fn fftfreq<F, I>(n: usize, d: F) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>,
    I: FloatIterable<F>
{
    let time = d * n.as_();
    (0..n).map(|i| i.as_() / time).collect()
}

/// Computes the frequency values associated fft based on `n` the length
/// of the collection and `d` the sampling period but the frequency values
/// are symmetric about the y-axis i.e. same the `scipy`'s `fftfreq` function
pub fn fftfreq_balanced<F, I>(n: usize, d: F) -> I 
where
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>, i32: AsPrimitive<F>,
    I: FloatIterable<F>
{
    let time = d * n.as_();
    let (pos_end, neg_start) = if n.is_even() {
        ((n as i32 / 2) -1 , -(n as i32 / 2))
    } else {
        ((n as i32 - 1) / 2, -(n as i32 - 1) / 2)
    };
    let pos_iter = 0..=pos_end;
    let neg_iter = neg_start..=-1;

    pos_iter.chain(neg_iter).map(|i| i.as_() / time).collect()
}


/// Wraps an angle in radians to the range (-π, π].
pub fn wrap_phase<F: Float + FloatConst + NumAssignOps>(angle: F) -> F {
    if angle >= F::PI() {
        angle - F::TAU()
    } else if angle <= -F::PI() {
        angle + F::TAU()
    } else {
        angle
    }
}

#[cfg(test)]
mod tests {
    
    use super::*;
    use crate::test_utils::{read_json, Data, Json};
    use crate::test_utils::{ self as test, test_dft, test_idft };
    use ndarray::prelude::*;

    const ATOL_F64: f64 = 1e-12;
    const RTOL_F64: f64 = 1e-9;

    // Really loose tolerances for f32 because we're checking complex numbers
    // which is more difficult, especially near zero where the phase can suddenly
    // jump by π for a small change in the real or imaginary part. Precision errors
    // for FFT algorithms can also accumulate. These values were found by trial-and-error.
    const ATOL_F32: f32 = 1e-1;
    const RTOL_F32: f32 = 1e-1;

    

    #[test]
    fn test_dft_vec_f32() {
        test_dft!(f32, Vec<f32>, Vec<Complex<f32>>, RTOL_F32, ATOL_F32);
    }

    #[test]
    fn test_dft_vec_f64() {
        test_dft!(f64, Vec<f64>, Vec<Complex<f64>>, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_dft_arr_f64() {
        test_dft!(f64, Array1<f64>, Array1<Complex<f64>>, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_dft_mix1_method_f64() {
        test_dft!(f64, Vec<f64>, Array1<Complex<f64>>, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_idft_vec_f32() {
        test_idft!(f32, Vec<Complex<f32>>, Vec<f32>, RTOL_F32, ATOL_F32);
    }

    #[test]
    fn test_idft_vec_f64() {
        test_idft!(f64, Vec<Complex<f64>>, Vec<f64>, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_idft_arr_f64() {
        test_idft!(f64, Array1<Complex<f64>>, Array1<f64>, RTOL_F64, ATOL_F64);
    }

   #[test]
    fn test_fftfreq_balanced() {
        let json_data: Json<f64> = read_json("datasets/fft/fftfreq/fftfreq.json");
        let (n, d) = match json_data.input_data {
             Data::FftFreqVals { n, d } => (n, d),
             _ => panic!("Read the input data incorrectly")
        };
        let scipy: Vec<f64> = match json_data.output_data {
        Data::<f64>::Array(output) => output,
        _ => panic!("Read the input data incorrectly")
        };

        let freqs: Vec<_> = fftfreq_balanced(n as usize, d);

        for (&f1, &f2) in freqs.iter().zip(scipy.iter()) {
            assert!(test::nearly_equal(f1, f2, RTOL_F64, ATOL_F64),
                "{} != {}", f1, f2);
        }
    }
}


