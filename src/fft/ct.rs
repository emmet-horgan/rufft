pub mod complex;
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, AsPrimitive, Zero, One};
use std::ops::IndexMut;
use std::iter::{ExactSizeIterator, Iterator, FromIterator, IntoIterator};
use crate::traits::{Collection, CollectionRef, FloatCollection, Iterable};

pub fn fft<C>(x: &C) -> C
where
    C: Iterable + FloatCollection + 
    FromIterator<Complex<C::FloatType>> + Clone + 
    IndexMut<usize, Output = Complex<C::FloatType>>,
    //C::FloatType: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<C::FloatType>,
{
    let n = x.len();
    let zero = C::FloatType::zero();
    let one = C::FloatType::one();
    let twopi = C::FloatType::TAU();

    if n == 1 {
        x.iter()
            .map(|&x| Complex::new(x, zero))
            .collect()
    } else {
        let wn = Complex::new(zero, -twopi / (n.as_()));
        let wn = wn.exp();
        let mut w = Complex::new(one, zero);

        let x_even: C = x.iter().step_by(2).cloned().collect();
        let x_odd: C = x.iter().skip(1).step_by(2).cloned().collect();

        let y_even: C = fft(&x_even);
        let y_odd: C = fft(&x_odd);

        let mut y = C::from_iter(std::iter::repeat(Complex::new(zero, zero)).take(n));

        for j in 0..(n / 2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + n / 2] = y_even[j] - tmp;
            w *= wn;
        }
        y
    }
}

//pub fn fft<I, C, F>(x: &I) -> C
//where
//    // Bound F to float types
//    F: Float + FloatConst + NumAssign + 'static,
//    // Bound I to to an iterable collection of F
//    I: FromIterator<F> + Clone,
//    for<'a> &'a I: IntoIterator<Item = &'a F>,
//    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
//    // Ensure a usize can be converted to F, ideally this can be removed
//    usize: AsPrimitive<F>,
//    // Bound C to a collection of Complex<F>
//    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
//{
//    let n = x.into_iter().len();
//    let zero = F::zero();
//    let one = F::one();
//    let twopi = F::TAU();
//
//    if n == 1 {
//        x.into_iter()
//            .map(|&x| Complex::new(x, zero))
//            .collect()
//    } else {
//        let wn = Complex::new(zero, -twopi / (n.as_()));
//        let wn = wn.exp();
//        let mut w = Complex::new(one, zero);
//
//        let x_even: I = x.into_iter().step_by(2).cloned().collect();
//        let x_odd: I = x.into_iter().skip(1).step_by(2).cloned().collect();
//
//        let y_even: C = fft(&x_even);
//        let y_odd: C = fft(&x_odd);
//
//        let mut y = C::from_iter(std::iter::repeat(Complex::new(zero, zero)).take(n));
//
//        for j in 0..(n / 2) {
//            let tmp = w * y_odd[j];
//            y[j] = y_even[j] + tmp;
//            y[j + n / 2] = y_even[j] - tmp;
//            w *= wn;
//        }
//        y
//    }
//}



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
    C: FromIterator<F> + IndexMut<usize, Output = F>,
{   
    let n = x.into_iter().len();
    let tmp: I = complex::ifft_internal(x);
    tmp.into_iter().map(|x| x.re / n.as_()).collect()
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