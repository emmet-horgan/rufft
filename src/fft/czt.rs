use num_complex::Complex;
use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive };
use std::ops::IndexMut;
use super::ct;
use crate::itertools::complex::zero_pad;

fn chirp_complex<I, F>(n: usize) -> I
where 
    F: Float + FloatConst + NumAssign + 'static,
    I: FromIterator<Complex<F>> + Clone,
    usize: AsPrimitive<F>
{
    (0..n).map(|i| {
        Complex::from_polar(F::one(), 
        (F::PI() * i.as_() * i.as_()) / n.as_())
    }).collect()
}

fn inverse_chirp_complex<I, F>(n: usize) -> I
where 
    F: Float + FloatConst + NumAssign + 'static,
    I: FromIterator<Complex<F>> + Clone,
    usize: AsPrimitive<F>
{
    (0..n).map(|i| {
        Complex::from_polar(F::one(), 
        (-F::one() * F::PI() * i.as_() * i.as_()) / n.as_())
    }).collect()
}

pub fn fft<I, C, F>(x: &I) -> C
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static + std::fmt::Debug,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>> + Clone,
    for<'a> &'a C: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator + DoubleEndedIterator,
{
    let n = x.into_iter().len();
    let m = (2 * n) -1;
    let fft_len = m.next_power_of_two(); // Just use cooley-tukey for now
    let zero_pad_len = fft_len - m + n;

    let a: C = inverse_chirp_complex::<C, F>(n)
        .into_iter()
        .zip(x.into_iter())
        .map(|(c, v)| c * v)
        .collect();
    let b: C = chirp_complex(n);
    let reflection: C = b.into_iter().skip(1).take(n - 1).rev().cloned().collect();

    let a = zero_pad(fft_len, &a).expect("Internal padding error which should be impossible !");
    let b = zero_pad(zero_pad_len, &b).expect("Internal padding error which should be impossible !");

    let b: C = b.into_iter().chain(reflection.into_iter()).cloned().collect();

    let afft = ct::complex::fft(&a);
    let bfft = ct::complex::fft(&b);
    let convolution: C = afft
        .into_iter()
        .zip(bfft.into_iter())
        .map(|(a, b)| a * b)
        .collect();
    let tmp: C = ct::complex::ifft(&convolution);
    let product: C = inverse_chirp_complex(n);
    tmp.into_iter()
        .zip(product.into_iter())
        .map(|(a, b)| a * b)
        .collect()  
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_fft;
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
    fn test_fft_ct_vec_func_f64() {
        test_fft!(Vec<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_arr_func_f64() {
        test_fft!(Array1<f64>, Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix1_func_f64() {
        test_fft!(Vec<f64>, Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix2_func_f64() {
        test_fft!(Array1<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_fft_ct_vec_func_f32() {
        test_fft!(Vec<f32>, Vec<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }
    #[test]
    fn test_fft_ct_arr_func_f32() {
        test_fft!(Array1<f32>, Array1<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }
    #[test]
    fn test_fft_ct_mix1_func_f32() {
        test_fft!(Vec<f32>, Array1<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }
    #[test]
    fn test_fft_ct_mix2_func_f32() {
        test_fft!(Array1<f32>, Vec<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }
}