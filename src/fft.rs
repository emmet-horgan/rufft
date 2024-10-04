#![allow(non_snake_case)]
use ndarray::prelude::*;
use num::{zero, One, Zero};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, FloatConst, Num, NumAssign, AsPrimitive, NumOps, NumAssignOps};
//use num_traits::cast::AsPrimitive;
use std::ops::{Deref, DivAssign, Index, IndexMut, Mul};
use std::iter::{ExactSizeIterator, Iterator, FromIterator, IntoIterator};
use crate::traits::FloatCollection;

/*
pub struct DftIter<'a, T, U>
where 
    U: FloatVal,
    T: SigVal<U>
{
    arr: &'a [T],
    cur: usize,
    marker: std::marker::PhantomData<U>
}

impl<'a, T, U> DftIter<'a, T, U> 
where 
    U: FloatVal,
    T: SigVal<U>
{
    pub fn new(x: &'a [T]) -> Self {
        DftIter {
            arr: x,
            cur: 0,
            marker: std::marker::PhantomData
        }
    }
}

impl<'a, T, U> Iterator for DftIter<'a, T, U>
where
    U: FloatVal,
    T: SigVal<U>,
    usize: AsPrimitive<U>
{
    type Item = Complex<U>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.arr.len() {
            return None;
        }

        let mut sum = Complex::<U>::zero();
        let N = self.arr.len();
        let twopi: U = (2).as_() * U::PI();
        let zero = U::zero();
        for (n, x) in self.arr.into_iter().enumerate() {
            let phase: Complex<U> = Complex::<U>::new(zero,
                 -(twopi * n.as_() * self.cur.as_()) / N.as_());
            sum += Complex::<U>::new(x.as_(), zero) * phase.exp();
        }
        self.cur += 1;
        Some(sum)
    }
}

impl<'a, T, U> ExactSizeIterator for DftIter<'a, T, U>
where 
    U: FloatVal,
    T: SigVal<U>,
    usize: AsPrimitive<U>
{
    fn len(&self) -> usize {
        self.arr.len()
    }
}

pub fn dft<T, U, C>(x: &[T]) -> C
where
    U: FloatVal,
    T: SigVal<U>,
    usize: AsPrimitive<U>,
    C: FromIterator<Complex<U>>
{

    let N = x.len();
    let twopi: U = (2).as_() * U::PI();
    let zero: U = U::zero();
    x.into_iter().enumerate().map(|(k, _)|{
        x.into_iter().enumerate().map(|(n, j)| {
            let phase = Complex::<U>::new(zero, -(twopi * n.as_() * k.as_()) / N.as_());
            Complex::<U>::new(j.as_(), zero) * phase.exp()
        }).sum()
    }).collect()
}

pub fn idft<T, U, C>(x: &[T]) -> C
where
    U: FloatVal,
    T: SigVal<U>,
    usize: AsPrimitive<U>,
    C: FromIterator<Complex<U>>
{

    let N = x.len();
    let twopi: U = (2).as_() * U::PI();
    let zero: U = U::zero();
    x.into_iter().enumerate().map(|(k, _)|{
        x.into_iter().enumerate().map(|(n, j)| {
            let phase = Complex::<U>::new(zero, (twopi * n.as_() * k.as_()) / N.as_());
            Complex::<U>::new(j.as_(), zero) * phase.exp()
        }).sum::<Complex<U>>() * Complex::<U>::new(U::one() / N.as_(), zero)
    }).collect()
}
    */
/*
pub fn fftfreq<U, T>(n: usize, p: U) -> Array1<T> 
where 
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>,
    Array1<T>: DivAssign<T>
{
    let zero = T::zero();
    let one = T::one();
    let two = T::one() + T::one();
    let n_t: T = n.as_();
    
    let part0 :Array1<T>;
    let part1 :Array1<T>;
    if n % 2 == 0 {
        part0  = Array1::<T>::range(zero, n_t / two, one);
        part1  = Array1::<T>::range(-n_t / two, -zero, one);
        
    }
    else {
        part0 = Array1::<T>::range(zero, (n_t - one) / two, one);
        part1 = Array1::<T>::range(-(n_t - one) / two, -zero, one);
    }

    let mut arr = ndarray::concatenate![Axis(0), part0, part1];
    
    arr /= p.as_() * n_t;
    return arr;
}
*/
/*
pub fn dtft<U, T>(x: Array1<U>) -> impl Fn(Array1<T>) -> Array1<Complex<T>>
where 
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>,
{
    move |samples: Array1<T>| -> Array1<Complex<T>> {
        let zero = T::zero();
        let mut y: Array1<Complex<T>> = Array1::<Complex<T>>::zeros(samples.len());

        for (index, &w) in samples.iter().enumerate() {
            let mut sum: Complex<T> = Complex::<T>::new(zero, zero);
            for n in 0..x.len() {
                let n_t: T = n.as_();
                let phase: Complex<T> = Complex::<T>::new(zero, -n_t * w);
                
                let arg: Complex<T> = Complex::<T>::new(x[n].as_(), zero) * phase.exp();
                
                sum += arg;
            }
            y[index] = sum;
        }
        y
    }
}
    */

pub trait Fft<F: Float + FloatConst + NumAssign + 'static>
where 
    Self: IntoIterator<Item = F> + FromIterator<F> + Clone,
    <Self as IntoIterator>::IntoIter: ExactSizeIterator,
    for<'a> &'a Self: IntoIterator<Item = &'a F>,
    for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
    usize: AsPrimitive<F>,
{   
    //type Type: Float + FloatConst + NumAssign + 'static;
    fn fft_ct<C>(&self) -> C
    where
        C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>
    {
        fft_ct::<Self, C, F>(self)
    }
}

pub fn fft_ct<I, C, F>(x: &I) -> C
where
    F: Float + FloatConst + NumAssign + 'static,
    I: FromIterator<F> + Clone,
    for<'a> &'a I: IntoIterator<Item = &'a F>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    usize: AsPrimitive<F>,
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
{
    let N = x.into_iter().len();
    let zero = F::zero();
    let one = F::one();
    let twopi = F::TAU();

    if N == 1 {
        x.into_iter()
            .map(|&x| Complex::new(x, zero))
            .collect()
    } else {
        let wn = Complex::new(zero, -twopi / (N.as_()));
        let wn = wn.exp();
        let mut w = Complex::new(one, zero);

        let x_even: I = x.into_iter().step_by(2).cloned().collect();
        let x_odd: I = x.into_iter().skip(1).step_by(2).cloned().collect();

        let y_even: C = fft_ct(&x_even);
        let y_odd: C = fft_ct(&x_odd);

        let mut y = C::from_iter(std::iter::repeat(Complex::new(zero, zero)).take(N));

        for j in 0..(N / 2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + N / 2] = y_even[j] - tmp;
            w *= wn;
        }
        y
    }
}

impl<C, F> Fft<F> for C
where 
    Self: IntoIterator<Item = F> + FromIterator<F> + Clone,
    <Self as IntoIterator>::IntoIter: ExactSizeIterator,
    for<'a> &'a Self: IntoIterator<Item = &'a F>,
    for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>
{}



pub fn zero_pad<T: Num + Copy>(x: &Array1<T>) -> Option<Array1<T>> 
{
    let zero = T::zero();
    let N = x.len();

    // Add a check for N = 2^{m}, zero pad if not. Only executed once regardsless of recursion.
    if (N != 0) && (N & (N - 1)) != 0 {
        let mut power = 1;
        while power < N {
            power <<= 1;
        }
        let mut y = Array1::<T>::zeros(power);
        for i in 0..N {
            y[i] = x[i];
        }
        for i in N..power {
            y[i] = zero;
        }
        return Some(y);
    }
    else {
        return None;
    }
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
    use crate::io::{read_json, Data};
    use crate::test_utils as test;
    use ndarray::prelude::*;

    macro_rules! test_fft_ct_func {
        ($I:ty, $C:ty, $F:ty) => {
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
                        assert!(test::nearly_equal(mag_calc, mag[i]), 
                            "[mag] {} != {}", mag_calc, mag[i]);
                        assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i]), 
                            "[phase] {} != {}", wrap_phase(phase_calc), phase[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    macro_rules! test_fft_ct_method {
        ($I:ty, $C:ty, $F:ty) => {
            let json_data = read_json("datasets/fft/fft/fft.json");
            let output: $C = match json_data.input_data {
                Data::<$F>::Array(input) => Into::<$I>::into(input).fft(),
                _ => panic!("Read the input data incorrectly")
            };
            match json_data.output_data {
                Data::ComplexVals { mag, phase} => {
                    for i in 0..mag.len() {
                        let mag_calc = output[i].norm();
                        let phase_calc = output[i].arg();
                        assert!(test::nearly_equal(mag_calc, mag[i]), 
                            "[mag] {} != {}", mag_calc, mag[i]);
                        assert!(test::nearly_equal(wrap_phase(phase_calc), phase[i]), 
                            "[phase] {} != {}", wrap_phase(phase_calc), phase[i]);
                    }
                }
                _ => panic!("Read the output data incorrectly")
            }
        };
    }

    //#[test]
    //fn test_fft_ct_vec_func_f32() {
    //    test_fft_ct_func!(Vec<f32>, Vec<Complex<f32>>, f32);
    //}
    #[test]
    fn test_fft_ct_vec_method_f64() {
        test_fft_ct_method!(Vec<f64>, Vec<Complex<f64>>, f64);
    }
    #[test]
    fn test_fft_ct_arr_method_f64() {
        test_fft_ct_method!(Array1<f64>, Array1<Complex<f64>>, f64);
    }
    #[test]
    fn test_fft_ct_mix1_method_f64() {
        test_fft_ct_method!(Vec<f64>, Array1<Complex<f64>>, f64);
    }
    #[test]
    fn test_fft_ct_mix2_method_f64() {
        test_fft_ct_method!(Array1<f64>, Vec<Complex<f64>>, f64);
    }

    #[test]
    fn test_fft_ct_vec_func_f64() {
        test_fft_ct_func!(Vec<f64>, Vec<Complex<f64>>, f64);
    }
    #[test]
    fn test_fft_ct_arr_func_f64() {
        test_fft_ct_func!(Array1<f64>, Array1<Complex<f64>>, f64);
    }
    #[test]
    fn test_fft_ct_mix1_func_f64() {
        test_fft_ct_func!(Vec<f64>, Array1<Complex<f64>>, f64);
    }
    #[test]
    fn test_fft_ct_mix2_func_f64() {
        test_fft_ct_func!(Array1<f64>, Vec<Complex<f64>>, f64);
    }
}


