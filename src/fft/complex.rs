use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive };
use num_complex::Complex;
use std::ops::IndexMut;

pub fn dft<F, I>(x: &I) -> I
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
    I: IndexMut<usize, Output = Complex<F>>,
{
    let n = x.into_iter().len();
    let zero = F::zero();
    //let complex_zero = Complex::new(zero, zero);
    let twopi = F::TAU();
    x.into_iter().enumerate().map(|(i, _)|{ // Change to a range of some kind
        x.into_iter().enumerate().map(|(j, &f)| {
            let phase = Complex::<F>::new(zero, -(twopi * j.as_() * i.as_()) / n.as_());
            f * phase.exp()
        }).sum()
    }).collect()
}

pub(crate) fn idft_internal<F, I>(x: &I) -> I
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
{
    let n = x.into_iter().len();
    let zero = F::zero();
    let twopi = F::TAU();
    x.into_iter().enumerate().map(|(i, _)|{
        x.into_iter().enumerate().map(|(j, &f)| {
            let phase = Complex::<F>::new(zero, (twopi * j.as_() * i.as_()) / n.as_());
            f * phase.exp()
        }).map(|v| v).sum::<Complex<F>>() / n.as_()
    }).collect() // Experiment with the internal function returning an iterator 
    // of some kind. Need to reduce the amount of collections
}

pub fn idft<F, I>(x: &I) -> I
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
{
    idft_internal(x)
}

#[cfg(test)]
mod tests {
    
    use super::*;
    use crate::test_utils::{ test_complex_dft };
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
        test_complex_dft!(f32, Vec<Complex<f32>>, RTOL_F32, ATOL_F32);
    }

    #[test]
    fn test_dft_vec_f64() {
        test_complex_dft!(f64, Vec<Complex<f64>>, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_dft_arr_f64() {
        test_complex_dft!(f64, Array1<Complex<f64>>, RTOL_F64, ATOL_F64);
    }
    
}