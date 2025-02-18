use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use crate::traits::{ ComplexFloatIterable, ComplexFloatIterableMut, ExtendableIterable };

/// Clone and pad the complex valued input collection with the complex floating 
/// point type `F` by the length `n`. Meaning the output collection will have a 
/// final length of `x.len() + n`
pub fn pad_clone<F, I>(x: &I, padding: Complex<F>, n: usize) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F>
{
    let pad_iter = core::iter::repeat(padding).take(n);
    I::from_iter(x.iter().cloned().chain(pad_iter))
}

/// Clone and pad the complex valued input collection with complex floating point 
/// value `F` to the nearest power of two length
pub fn pad_to_nearest_power_of_two_clone<F, I>(x: &I, padding: Complex<F>) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F> + Clone,
{
    let n = x.iter().len();
    if n.is_power_of_two() {
        x.clone()
    } else {
        pad_clone(x, padding, n.next_power_of_two() - n) // Pad to the nearest power of 2
    }
}

/// Clone and zero pad the complex valued input collection by the length `n`. 
/// Meaning the output collection will have a final length of `x.len() + n`
pub fn zero_pad_clone<F, I>(x: &I, n: usize) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F>
{
    pad_clone(x, Complex::new(F::zero(), F::zero()), n)
}

/// Clone and zero pad the complex valued input collection to the nearest power 
/// of two length
pub fn zero_pad_to_nearest_power_of_two_clone<F, I>(x: &I) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterable<F> + Clone
{
    pad_to_nearest_power_of_two_clone(x, Complex::new(F::zero(), F::zero()))
}

/// Pad the complex valued input collection inplace with the complex floating 
/// point type `Complex<F>` by the length `n`. Meaning the collection will have a 
/// final length of `x.len() + n`
pub fn pad_inplace<F, I>(x: &mut I, padding: Complex<F>, n: usize) 
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterableMut<F> + ExtendableIterable
{
    let pad_iter = core::iter::repeat(padding).take(n);
    for p in pad_iter {
        x.push(p)
    }
}

/// Zero pad the complex valued input collection by the length `n`. Meaning 
/// the collection will have a final length of `x.len() + n`
pub fn zero_pad_inplace<F, I>(x: &mut I, n: usize)
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterableMut<F> + ExtendableIterable
{
    pad_inplace(x, Complex::new(F::zero(), F::zero()), n)
}

/// Pad the complex valued input collection in place with the complex floating 
/// point value `Complex<F>` to the nearest power of two length.
pub fn pad_to_nearest_power_of_two_inplace<F, I>(x: &mut I, padding: Complex<F>)
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterableMut<F> + ExtendableIterable
{
    let n = x.iter().len();
    if !n.is_power_of_two() {
        pad_inplace(x, padding, n.next_power_of_two() - n) // Pad to the nearest power of 2
    }
}

/// Zero pad the real valued input collection inplace to the nearest power 
/// of two length. Meaning the collection will have a final length of 
/// `x.len() + len`
pub fn zero_pad_to_nearest_power_of_two_inplace<F, I>(x: &mut I)
where
    F: Float + FloatConst + NumAssign + 'static,
    I: ComplexFloatIterableMut<F> + ExtendableIterable
{
    pad_to_nearest_power_of_two_inplace(x, Complex::new(F::zero(), F::zero()))
}