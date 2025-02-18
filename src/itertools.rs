//! Signal processing functions which operate on `Iterable` implemenentors. Named 
//! `itertools` after the `itertool` crate

pub mod complex;

use num_traits::{ Float, FloatConst, NumAssign };
use crate::traits::{ FloatIterable, FloatIterableMut, ExtendableIterable };

/// Clone and pad the real valued input collection with the floating 
/// point type `F` by the length `n`. Meaning the output collection will have a 
/// final length of `x.len() + n`
pub fn pad_clone<F, I>(x: &I, padding: F, n: usize) -> I 
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterable<F>
{
    let pad_iter = core::iter::repeat(padding).take(n);
    I::from_iter(x.iter().cloned().chain(pad_iter))
}

/// Clone and zero pad the real valued input collection by the length `n`.
/// Meaning the output collection will have a  final length of `x.len() + n`
pub fn zero_pad_clone<F, I>(x: &I, n: usize) -> I 
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterable<F>
{
    pad_clone(x, F::zero(), n)
}

/// Clone and pad the real valued input collection with floating point value `F`
/// to the nearest power of two length
pub fn pad_to_nearest_power_of_two_clone<F, I>(x: &I, padding: F) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterable<F> + Clone
{
    let n = x.iter().len();
    if n.is_power_of_two() {
        x.clone()
    } else {
        pad_clone(x, padding, n.next_power_of_two() - n) // Pad to the nearest power of 2
    }
}

/// Clone and zero pad the real valued input collection to the nearest power 
/// of two length
pub fn zero_pad_to_nearest_power_of_two_clone<F, I>(x: &I) -> I
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterable<F> + Clone
{
    pad_to_nearest_power_of_two_clone(x, F::zero())
}

/// Pad the real valued input collection inplace with the floating 
/// point type `F` by the length `n`. Meaning the collection will have a 
/// final length of `x.len() + n`
pub fn pad_inplace<F, I>(x: &mut I, padding: F, n: usize) 
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterableMut<F> + ExtendableIterable
{
    let pad_iter = core::iter::repeat(padding).take(n);
    for p in pad_iter {
        x.push(p)
    }
}

/// Zero pad the real valued input collection by the length `n`. Meaning 
/// the collection will have a final length of `x.len() + n`
pub fn zero_pad_inplace<F, I>(x: &mut I, n: usize)
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterableMut<F> + ExtendableIterable
{
    pad_inplace(x, F::zero(), n)
}

/// Pad the real valued input collection in place with floating point value `F`
/// to the nearest power of two length
pub fn pad_to_nearest_power_of_two_inplace<F, I>(x: &mut I, padding: F)
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterableMut<F> + ExtendableIterable
{
    let n = x.iter().len();
    if !n.is_power_of_two() {
        pad_inplace(x, padding, n.next_power_of_two() - n) // Pad to the nearest power of 2
    }
}

/// Zero pad the real valued input collection inplace to the nearest power 
/// of two length
pub fn zero_pad_to_nearest_power_of_two_inplace<F, I>(x: &mut I)
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FloatIterableMut<F> + ExtendableIterable
{
    pad_to_nearest_power_of_two_inplace(x, F::zero())
}
