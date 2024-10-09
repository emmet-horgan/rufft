use num::Integer;
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, AsPrimitive, NumAssignOps};
use std::ops::IndexMut;
use std::iter::{ExactSizeIterator, Iterator, FromIterator, IntoIterator};
use crate::traits::{Collection, CollectionRef};


pub fn dtft<I, C, F>(x: I) -> impl Fn(I) -> C
where 
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
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

fn idft_internal<I, F>(x: &I) -> I
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


pub fn idft<I, C, F>(x: &I) -> C
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
    C: FromIterator<F> + IndexMut<usize, Output = F>,
{
    idft_internal(x).into_iter().map(|x| x.re).collect()
}

pub fn idft_complex<I, F>(x: &I) -> I
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

pub fn dft<I, C, F>(x: &I) -> C
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
{
    let n = x.into_iter().len();
    let zero = F::zero();
    let twopi = F::TAU();
    x.into_iter().enumerate().map(|(i, _)|{ // Change to a range of some kind
        x.into_iter().enumerate().map(|(j, &f)| {
            let phase = Complex::<F>::new(zero, -(twopi * j.as_() * i.as_()) / n.as_());
            Complex::<F>::new(f, zero) * phase.exp()
        }).sum()
    }).collect()
}

pub fn dft_complex<I, F>(x: &I) -> I
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

pub fn fft_ct<I, C, F>(x: &I) -> C
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    // Ensure a usize can be converted to F, ideally this can be removed
    usize: AsPrimitive<F>,
    // Bound C to a collection of Complex<F>
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
{
    let n = x.into_iter().len();
    let zero = F::zero();
    let one = F::one();
    let twopi = F::TAU();

    if n == 1 {
        x.into_iter()
            .map(|&x| Complex::new(x, zero))
            .collect()
    } else {
        let wn = Complex::new(zero, -twopi / (n.as_()));
        let wn = wn.exp();
        let mut w = Complex::new(one, zero);

        let x_even: I = x.into_iter().step_by(2).cloned().collect();
        let x_odd: I = x.into_iter().skip(1).step_by(2).cloned().collect();

        let y_even: C = fft_ct(&x_even);
        let y_odd: C = fft_ct(&x_odd);

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

pub fn fft_ct_complex<I, F>(x: &I) -> I
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

        let y_even: I = fft_ct_complex(&x_even);
        let y_odd: I = fft_ct_complex(&x_odd);

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

fn ifft_ct_internal<I, F>(x: &I) -> I
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

        let y_even: I = ifft_ct_internal(&x_even);
        let y_odd: I = ifft_ct_internal(&x_odd);

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

pub fn ifft_ct<I, C, F>(x: &I) -> C
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
    let tmp: I = ifft_ct_internal(x);
    tmp.into_iter().map(|x| x.re / n.as_()).collect()
}

pub fn ifft_ct_complex<I, C, F>(x: &I) -> C
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
    let tmp: I = ifft_ct_internal(x);
    tmp.into_iter().map(|x| x / n).collect()
}



pub fn fftfreq<I, F>(n: usize, d: F) -> I
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
{
    let time = d * n.as_();
    (0..n).map(|i| i.as_() / time).collect()
}

pub fn fftfreq_balanced<I, F>(n: usize, d: F) -> I 
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>,
    i32: AsPrimitive<F>,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
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

pub fn pad<I, F>(x: &I, padding: F, len: usize) -> Result<I, ()> 
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
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

pub fn pad_complex<I, F>(x: &I, padding: Complex<F>, len: usize) -> Result<I, ()> 
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

pub fn pad_to_nearest_power_of_two_complex<I, F>(x: &I, padding: Complex<F>) -> Result<I, ()>
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
        Ok(pad_complex(x, padding, n.next_power_of_two())?) // Pad to the nearest power of 2
    }
}

pub fn zero_pad_complex<I, F>(n: usize, x: &I) -> Result<I, ()> 
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    Ok(pad_complex(x, Complex::new(F::zero(), F::zero()), n)?)
}

pub fn zero_pad_to_nearest_power_of_two_complex<I, F>(x: &I) -> Result<I, ()>
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<Complex<F>> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a Complex<F>>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    pad_to_nearest_power_of_two_complex(x, Complex::new(F::zero(), F::zero()))
}

pub fn zero_pad<I, F>(n: usize, x: &I) -> Result<I, ()> 
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    Ok(pad(x, F::zero(), n)?)
}


pub fn pad_to_nearest_power_of_two<I, F>(x: &I, padding: F) -> Result<I, ()>
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    let n = x.into_iter().len();
    if n.is_power_of_two() {
        Ok(x.clone())
    } else {
        Ok(pad(x, padding, n.next_power_of_two())?) // Pad to the nearest power of 2
    }
}

pub fn zero_pad_to_nearest_power_of_two<I, F>(x: &I) -> Result<I, ()>
where
    // Bound F to float types
    F: Float + FloatConst + NumAssign + 'static,
    // Bound I to to an iterable collection of F
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
{
    pad_to_nearest_power_of_two(x, F::zero())
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

pub fn chirp_complex<I, F>(n: usize) -> I
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

pub fn inverse_chirp_complex<I, F>(n: usize) -> I
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

pub fn fft_bluestein<I, C, F>(x: &I) -> C
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

    let a = zero_pad_complex(fft_len, &a).expect("Internal padding error which should be impossible !");
    let b = zero_pad_complex(zero_pad_len, &b).expect("Internal padding error which should be impossible !");

    let b: C = b.into_iter().chain(reflection.into_iter()).cloned().collect();

    let afft = fft_ct_complex(&a);
    let bfft = fft_ct_complex(&b);
    let convolution: C = afft
        .into_iter()
        .zip(bfft.into_iter())
        .map(|(a, b)| a * b)
        .collect();
    let tmp: C = ifft_ct_complex(&convolution);
    let product: C = inverse_chirp_complex(n);
    tmp.into_iter()
        .zip(product.into_iter())
        .map(|(a, b)| a * b)
        .collect()  
}

#[macro_export]
macro_rules! fft {
    ($x:expr) => {
        match fft::zero_pad(&$x) {
            None => {
                fft::fft_ct(&$x)
            },
            Some(x) => {
                fft::dft::<f64 ,f64>(&$x)
            }
        }
    };

    ($x:expr, $cfg:expr) => {
        match fft::zero_pad(&$x) {
            None => {
                fft::fft_ct(&$x)
            },
            Some(x) => {
                let tmp = fft::fft_ct(&x);
                tmp.slice(s![..$x.len()]).to_owned() // Maybe remove this line ?
            }
        }
        
    }
}

#[macro_export]
macro_rules! ifft {
    ($x:expr) => {
        std::unimplemented!();
    };
}

#[macro_export]
macro_rules! fftfreq {
    ($x:expr, $t:expr) => {
        match fft::zero_pad(&$x) {
            None => {
                fft::fftfreq($x.len(), $t)
            }
            Some(x) => {
                fft::fftfreq(x.len(), $t) // Not sure if this is correct
            }
        }
    };
}

#[cfg(test)]
mod tests {
    
    use super::*;
    use crate::traits::Fft; // For methods
    use crate::io::{read_json, Data, Json};
    use crate::test_utils as test;
    use ndarray::prelude::*;

    const ATOL_F64: f64 = 1e-12;
    const RTOL_F64: f64 = 1e-9;

    // Really loose tolerances for f32 because we're checking complex numbers
    // which is more difficult, especially near zero where the phase can suddenly
    // jump by π for a small change in the real or imaginary part. Precision errors
    // for FFT algorithms can also accumulate. These values were found by trial-and-error.
    const ATOL_F32: f32 = 1e-1;
    const RTOL_F32: f32 = 1e-1;

    macro_rules! test_dft_fft_ct {
        ($I:ty, $C:ty, $F:ty, $rtol:expr, $atol:expr) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let fft_output: $C;
            let dft_output: $C;
            match json_data.input_data {
                Data::<$F>::Array(input) => {
                    fft_output = fft_ct::<$I, $C, $F>(&(input.clone().into()));
                    dft_output = dft::<$I, $C, $F>(&(input.into()));
                },
                _ => panic!("Read the input data incorrectly")
            };
            for (dft_comp, fft_comp) in std::iter::zip(dft_output.iter(), fft_output.iter()) {
                assert!(test::nearly_equal_complex(*dft_comp, *fft_comp, $rtol, $atol), 
                    "complex[{} != {}]\nmag[{} != {}]\nphase[{} != {}]", 
                    dft_comp, fft_comp, dft_comp.norm(), fft_comp.norm(), dft_comp.arg(), fft_comp.arg());
            }
        }
    }

    macro_rules! test_idft_func {
        ($I:ty, $C:ty, $F:ty, $rtol:expr, $atol:expr) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.output_data {
                Data::ComplexVals { mag, phase } => {
                    let input: $I = std::iter::zip(mag, phase).map(|(m, p)| Complex::from_polar(m, p)).collect();
                    idft::<$I, $C, $F>(&(input.into()))
                },
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.input_data {
                Data::<$F>::Array(input) => {
                    for i in 0..input.len() {
                        assert!(test::nearly_equal(output[i], input[i], $rtol, $atol), 
                            "{} != {}", output[i], input[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    macro_rules! test_ifft_ct_func {
        ($I:ty, $C:ty, $F:ty, $rtol:expr, $atol:expr) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.output_data {
                Data::ComplexVals { mag, phase } => {
                    let input: $I = std::iter::zip(mag, phase).map(|(m, p)| Complex::from_polar(m, p)).collect();
                    ifft_ct::<$I, $C, $F>(&(input.into()))
                },
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.input_data {
                Data::<$F>::Array(input) => {
                    for i in 0..input.len() {
                        assert!(test::nearly_equal(output[i], input[i], $rtol, $atol), 
                            "{} != {}", output[i], input[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    macro_rules! test_fft_ct_func {
        ($I:ty, $C:ty, $F:ty, $rtol:expr, $atol:expr) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.input_data {
                Data::<$F>::Array(input) => fft_ct::<$I, $C, $F>(&(input.into())),
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.output_data {
                Data::ComplexVals { mag, phase} => {
                    for i in 0..mag.len() {
                        let mag_calc = output[i].norm();
                        let phase_calc = output[i].arg();
                        assert!(test::nearly_equal(mag_calc, mag[i], $rtol, $atol), 
                            "[mag] {} != {}", mag_calc, mag[i]);
                        assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i], $rtol, $atol), 
                            "[phase] {} != {}", wrap_phase(phase_calc), phase[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    macro_rules! test_fft_bluestein_func {
        ($I:ty, $C:ty, $F:ty, $rtol:expr, $atol:expr) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.input_data {
                Data::<$F>::Array(input) => fft_bluestein::<$I, $C, $F>(&(input.into())),
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.output_data {
                Data::ComplexVals { mag, phase} => {
                    for i in 0..mag.len() {
                        let mag_calc = output[i].norm();
                        let phase_calc = output[i].arg();
                        assert!(test::nearly_equal(mag_calc, mag[i], $rtol, $atol), 
                            "{}: [mag] {} != {}", i, mag_calc, mag[i]);
                        assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i], $rtol, $atol), 
                            "{}: [phase] {} != {}", i, wrap_phase(phase_calc), phase[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    macro_rules! test_fft_ct_method {
        ($I:ty, $C:ty, $F:ty, $rtol:expr, $atol:expr) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.input_data {
                Data::<$F>::Array(input) => Into::<$I>::into(input).fft_ct(),
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.output_data {
                Data::ComplexVals { mag, phase} => {
                    for i in 0..mag.len() {
                        let mag_calc = output[i].norm();
                        let phase_calc = output[i].arg();
                        assert!(test::nearly_equal(mag_calc, mag[i], $rtol, $atol), 
                            "[mag] {} != {}", mag_calc, mag[i]);
                        assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i], $rtol, $atol), 
                            "[phase] {} != {}", wrap_phase(phase_calc), phase[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    macro_rules! test_dft_func {
        ($I:ty, $C:ty, $F:ty) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.input_data {
                Data::<$F>::Array(input) => dft::<$I, $C, $F>(&(input.into())),
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.output_data {
                Data::ComplexVals { mag, phase} => {
                    for i in 0..mag.len() {
                        let mag_calc = output[i].norm();
                        let phase_calc = output[i].arg();
                        assert!(test::nearly_equal(mag_calc, mag[i]), 
                            "[mag] {} != {}", mag_calc, mag[i]);
                        assert!(test::nearly_equal(wrap_phase(phase_calc) - <$F>::PI(), phase[i]), 
                            "[phase] {} != {}", phase_calc - <$F>::PI(), phase[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    #[test]
    fn test_macro_dft_vec_func_f32() {
        test_dft_fft_ct!(Vec<f32>, Vec<Complex<f32>>, f32, RTOL_F32, ATOL_F32);
    }

    #[test]
    fn test_macro_dft_vec_func_f64() {
        test_dft_fft_ct!(Vec<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_fft_ct_vec_method_f64() {
        test_fft_ct_method!(Vec<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_arr_method_f64() {
        test_fft_ct_method!(Array1<f64>, Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix1_method_f64() {
        test_fft_ct_method!(Vec<f64>, Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix2_method_f64() {
        test_fft_ct_method!(Array1<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_fft_ct_vec_func_f64() {
        test_fft_ct_func!(Vec<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_arr_func_f64() {
        test_fft_ct_func!(Array1<f64>, Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix1_func_f64() {
        test_fft_ct_func!(Vec<f64>, Array1<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
    }
    #[test]
    fn test_fft_ct_mix2_func_f64() {
        test_fft_ct_func!(Array1<f64>, Vec<Complex<f64>>, f64, RTOL_F64, ATOL_F64);
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

        let freqs = fftfreq_balanced::<Vec<f64>, f64>(n as usize, d);

        for (&f1, &f2) in freqs.iter().zip(scipy.iter()) {
            assert!(test::nearly_equal(f1, f2, RTOL_F64, ATOL_F64),
                "{} != {}", f1, f2);
        }
    }
    #[test]
    fn test_dft_complex_func_vec_f64() {
        let json_data = read_json("datasets/fft/complex_fft/complex_fft.json");
        let output: Vec<Complex<f64>> = match json_data.input_data {
            Data::ComplexVals { mag, phase } => {
                let input: Vec<Complex<f64>> = std::iter::zip(mag, phase).map(|(m, p)| Complex::from_polar(m, p)).collect();
                dft_complex::<Vec<Complex<f64>>, f64>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.output_data {
            Data::ComplexVals { mag, phase} => {
                for i in 0..mag.len() {
                    let mag_calc = output[i].norm();
                    let phase_calc = output[i].arg();
                    assert!(test::nearly_equal(mag_calc, mag[i], 1e-6, 1e-9), 
                        "[mag] {} != {}", mag_calc, mag[i]);
                    assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i], 1e-6, 1e-9), 
                        "[phase] {} != {}", wrap_phase(phase_calc), phase[i]);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    }
    #[test]
    fn test_idft_complex_func_vec_f64() {
        let json_data = read_json("datasets/fft/complex_fft/complex_fft.json");
        let input: Vec<Complex<f64>> = match json_data.output_data {
            Data::ComplexVals { mag, phase } => {
                let input: Vec<Complex<f64>> = std::iter::zip(mag, phase).map(|(m, p)| Complex::from_polar(m, p)).collect();
                idft_complex::<Vec<Complex<f64>>, f64>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.input_data {
            Data::ComplexVals { mag, phase} => {
                for i in 0..mag.len() {
                    let mag_calc = input[i].norm();
                    let phase_calc = input[i].arg();
                    assert!(test::nearly_equal(mag_calc, mag[i], 1e-6, 1e-9), 
                        "[mag] {} != {}", mag_calc, mag[i]);
                    assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i], 1e-6, 1e-9), 
                        "[phase] {} != {}", wrap_phase(phase_calc), phase[i]);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    }
    #[test]
    fn test_fft_bluestein_func() {
        //test_fft_bluestein_func!(Vec<f64>, Vec<Complex<f64>>, f64, 1e-6, 1e-9);
        let json_data = read_json("datasets/fft/fft/fft.json");
        let output: Vec<Complex<f64>> = match json_data.input_data {
            Data::<f64>::Array(input) => fft_bluestein::<Vec<f64>, Vec<Complex<f64>>, f64>(&(input.into())),
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.output_data {
            Data::ComplexVals { mag, phase} => {
                for i in 0..mag.len() {
                    let reference = Complex::from_polar(mag[i], phase[i]);
                    // Reduce tolerance a little bit because bluestein requires two ffts and an ifft
                    // which will accumulate more precision errors
                    assert!(test::nearly_equal_complex(output[i], reference, RTOL_F64, ATOL_F64), 
                        "{} => {} != {}", i, output[i], reference);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    }
    #[test]
    fn test_idft_func() {
        test_idft_func!(Vec<Complex<f64>>, Vec<f64>, f64, RTOL_F64, ATOL_F64);
    }

    #[test]
    fn test_ifft_ct_func() {
        test_ifft_ct_func!(Vec<Complex<f64>>, Vec<f64>, f64, RTOL_F64, ATOL_F64);
    }
}


