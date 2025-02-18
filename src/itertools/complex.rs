use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use crate::traits::ComplexFloatIterable;

/// Clone and pad the complex valued input collection with the floating 
/// point type `F` to the length `len`
pub fn pad<F, I>(x: &I, padding: Complex<F>, len: usize) -> Result<I, ()> 
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F>
{
    let n = x.iter().len();
    if len < n {
        return Err(());
    }
    let num_padding = len - n;
    let pad_iter = core::iter::repeat(padding).take(num_padding);
    Ok(I::from_iter(x.iter().cloned().chain(pad_iter)))
}

/// Clone and pad the complex valued input collection with floating point value `F`
/// to the nearest power of two length
pub fn pad_to_nearest_power_of_two<F, I>(x: &I, padding: Complex<F>) -> Result<I, ()>
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F> + Clone,
{
    let n = x.iter().len();
    if n.is_power_of_two() {
        Ok(x.clone())
    } else {
        Ok(pad(x, padding, n.next_power_of_two())?) // Pad to the nearest power of 2
    }
}

/// Clone and zero pad the complex valued input collection to the length `len`
pub fn zero_pad<F, I>(n: usize, x: &I) -> Result<I, ()> 
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F>
{
    Ok(pad(x, Complex::new(F::zero(), F::zero()), n)?)
}

/// Clone and zero pad the complex valued input collection to the nearest power 
/// of two length
pub fn zero_pad_to_nearest_power_of_two<F, I>(x: &I) -> Result<I, ()>
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F> + Clone
{
    pad_to_nearest_power_of_two(x, Complex::new(F::zero(), F::zero()))
}