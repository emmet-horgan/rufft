use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};

pub fn pad<I, F>(x: &I, padding: Complex<F>, len: usize) -> Result<I, ()> 
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    let n = x.into_iter().len();
    if len < n {
        return Err(());
    }
    let num_padding = len - n;
    let pad_iter = std::iter::repeat(padding).take(num_padding);
    Ok(I::from_iter(x.into_iter().cloned().chain(pad_iter)))
}

pub fn pad_to_nearest_power_of_two<I, F>(x: &I, padding: Complex<F>) -> Result<I, ()>
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    let n = x.into_iter().len();
    if n.is_power_of_two() {
        Ok(x.clone())
    } else {
        Ok(pad(x, padding, n.next_power_of_two())?) // Pad to the nearest power of 2
    }
}

pub fn zero_pad<I, F>(n: usize, x: &I) -> Result<I, ()> 
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    Ok(pad(x, Complex::new(F::zero(), F::zero()), n)?)
}

pub fn zero_pad_to_nearest_power_of_two<I, F>(x: &I) -> Result<I, ()>
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    pad_to_nearest_power_of_two(x, Complex::new(F::zero(), F::zero()))
}