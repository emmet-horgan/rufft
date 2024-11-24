pub mod complex;
use num_complex::Complex;
use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive };
use core::ops::IndexMut;
use itertools::izip;
use crate::traits::Iterable;

/// Computes the cooley-tukey fast fourier transform of the real input collection
pub fn fft<F, I, C>(x: &I) -> C
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> I: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    for<'c> C: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c Complex<F>>,
    C: IndexMut<usize, Output = Complex<F>>,
    usize: AsPrimitive<F>,
{
    let n = x.len();
    let zero = F::zero();
    let one = F::one();
    let twopi = F::TAU();

    if n == 1 {
        x.iter()
            .map(|&x| Complex::new(x, zero))
            .collect()
    } else {
        let wn = Complex::new(zero, -twopi / (n.as_()));
        let wn = wn.exp();
        let mut w = Complex::new(one, zero);

        let x_even: I = x.iter().step_by(2).cloned().collect();
        let x_odd: I = x.iter().skip(1).step_by(2).cloned().collect();

        let y_even: C = fft(&x_even);
        let y_odd: C = fft(&x_odd);

        let mut y = C::from_iter(core::iter::repeat(Complex::new(zero, zero)).take(n));
        izip!(y_even.iter(), y_odd.iter())
            .enumerate()
            .for_each(|(j, (even, odd))| {
                let tmp = w * odd;
                y[j] = even + tmp;
                y[j + n / 2] = even - tmp;
                w *= wn;
            });
        y
    }
}


/// Compute the inverse cooley-tukey transform of the complex input collection and returns
/// the real valued output collection
/// The output *is* normalized.
pub fn ifft<F, I, C>(x: &I) -> C
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> I: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c Complex<F>>,
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    I: IndexMut<usize, Output = Complex<F>>,
    usize: AsPrimitive<F>,
{   
    let n = x.len();
    complex::ifft_internal(x).iter().map(|x| x.re / n.as_()).collect()
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
        test_fft!(f64, Vec<f64>, Vec<Complex<f64>>, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_arr_func_f64() {
        test_fft!(f64, Array1<f64>, Array1<Complex<f64>>, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix1_func_f64() {
        test_fft!(f64, Vec<f64>, Array1<Complex<f64>>, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix2_func_f64() {
        test_fft!(f64, Array1<f64>, Vec<Complex<f64>>, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_fft_ct_vec_func_f32() {
        test_fft!(f32, Vec<f32>, Vec<Complex<f32>>, RTOL_F32, ATOL_F32);
    }
    #[test]
    fn test_fft_ct_arr_func_f32() {
        test_fft!(f32, Array1<f32>, Array1<Complex<f32>>, RTOL_F32, ATOL_F32);
    }
    #[test]
    fn test_fft_ct_mix1_func_f32() {
        test_fft!(f32, Vec<f32>, Array1<Complex<f32>>, RTOL_F32, ATOL_F32);
    }
    #[test]
    fn test_fft_ct_mix2_func_f32() {
        test_fft!(f32, Array1<f32>, Vec<Complex<f32>>, RTOL_F32, ATOL_F32);
    }
}