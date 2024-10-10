use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, AsPrimitive, NumAssignOps};
use std::ops::IndexMut;
use std::iter::{ExactSizeIterator, Iterator, FromIterator, IntoIterator};
use crate::traits::{Collection, CollectionRef};

pub fn fft<I, F>(x: &I) -> I
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone + IndexMut<usize, Output = Complex<F>>,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
{
    let n = x.into_iter().len();
    let n_f: F = n.as_();
    let zero = Complex::new(F::zero(), F::zero());
    let one_real = Complex::new(F::one(), F::zero());
    //let one_complex = Complex::new(F::zero(), F::one());
    //let twopi_real = Complex::new(F::TAU(), F::zero());
    let twopi_complex = Complex::new(F::zero(), F::TAU());
    if n == 1 {
        x.into_iter().cloned().collect()
    } else {
        let wn = -twopi_complex / n_f;
        //let wn = Complex::new(zero, -twopi / (n.as_()));
        let wn = wn.exp();
        let mut w = one_real;
        let x_even: I = x.into_iter().step_by(2).cloned().collect();
        let x_odd: I = x.into_iter().skip(1).step_by(2).cloned().collect();
        let y_even: I = fft(&x_even);
        let y_odd: I = fft(&x_odd);
        let mut y = I::from_iter(std::iter::repeat(zero).take(n));
        for j in 0..(n / 2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + n / 2] = y_even[j] - tmp;
            w *= wn;
        }
        y
    }
}

pub(crate) fn ifft_internal<I, F>(x: &I) -> I
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone + IndexMut<usize, Output = Complex<F>>,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
    //C: FromIterator<F> + IndexMut<usize, Output = F>,
{
    let n = x.into_iter().len();
    let zero = F::zero();
    let one = F::one();
    let twopi = F::TAU();
    if n == 1 {
        x.into_iter().cloned().collect()
    } else {
        let wn = Complex::new(zero, twopi / (n.as_()));
        let wn = wn.exp();
        let mut w = Complex::new(one, zero);
        let x_even: I = x.into_iter().step_by(2).cloned().collect();
        let x_odd: I = x.into_iter().skip(1).step_by(2).cloned().collect();
        let y_even: I = ifft_internal(&x_even);
        let y_odd: I = ifft_internal(&x_odd);
        let mut y = I::from_iter(std::iter::repeat(Complex::new(zero, zero)).take(n));
        for j in 0..(n / 2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + n / 2] = y_even[j] - tmp;
            w *= wn;
        }
        y
    }
}

pub fn ifft<I, C, F>(x: &I) -> C
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone + IndexMut<usize, Output = Complex<F>>,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
{   
    let n: F = x.into_iter().len().as_();
    let tmp: I = ifft_internal(x);
    tmp.into_iter().map(|x| x / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{ test_complex_fft, test_complex_ifft};
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
        test_complex_fft!(Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_fft_ct_arr_func_f64() {
        test_complex_fft!(Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_fft_ct_vec_func_f32() {
        test_complex_fft!(Vec<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }

    #[test]
    fn test_fft_ct_arr_func_f32() {
        test_complex_fft!(Array1<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }
    
    #[test]
    fn test_ifft_vec_f64() {
        test_complex_ifft!(Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_ifft_vec_f32() {
        test_complex_ifft!(Vec<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }

    #[test]
    fn test_ifft_arr_f64() {
        test_complex_ifft!(Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_ifft_arr_f32() {
        test_complex_ifft!(Array1<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }
    
}