#![allow(non_snake_case)]
use ndarray::prelude::*;
use num::{zero, One, Zero};
use num_complex::{Complex, ComplexFloat};
use crate::traits::{FloatVal, SigVal};
use num_traits::{Float, FloatConst, Num, NumAssign, AsPrimitive, NumOps, NumAssignOps};
//use num_traits::cast::AsPrimitive;
use std::ops::{Deref, DivAssign, Index, IndexMut, Mul};
use std::iter::{ExactSizeIterator, Iterator, FromIterator, IntoIterator};

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

pub trait TyEq
where
    Self: From<Self::Type> + Into<Self::Type>,
    Self::Type: From<Self> + Into<Self>,
{
    type Type;
}

impl<T> TyEq for T {
    type Type = T;
}

pub trait HasInnerFloat {
    type InnerFloat: Float + FloatConst + NumAssign + 'static;
}   

impl<T: 'static + Float + FloatConst + NumAssign> HasInnerFloat for Complex<T> {
    type InnerFloat = T;
}

pub trait Fft
where 
    Self: IntoIterator + FromIterator<<Self as IntoIterator>::Item> + IndexMut<usize, Output = <Self as IntoIterator>::Item>,
    Self: Deref<Target = [<Self as IntoIterator>::Item]> + Clone,
    <Self as IntoIterator>::Item: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<<Self as IntoIterator>::Item>
{
    fn fft_ct<C>(&self) -> C
    where
        C: IntoIterator<Item = Complex<<Self as IntoIterator>::Item>> + FromIterator<Complex<<Self as IntoIterator>::Item>> + IndexMut<usize, Output = Complex<<Self as IntoIterator>::Item>>
    {
        fft_ct::<Self, C, <Self as IntoIterator>::Item>(&self)
    }
}

pub fn fft_ct<I, C, F>(x: &I) -> C
where 
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>,
    C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
    I: FromIterator<F> + Deref<Target = [F]>,
{
    let N = x.len();
    let zero = F::zero();
    let one = F::one();
    let twopi = F::TAU();

    if N == 1 {
        x.iter().map(|x| Complex::<F>::new(*x, zero)).collect()
    } else {
        let wn = Complex::<F>::new(zero, -twopi / N.as_());
        let wn = wn.exp();
        let mut w = Complex::<F>::new(one, zero);

        let x_even: I = x.iter().step_by(2).map(|k| *k).collect();
        let x_odd: I = x.iter().skip(1).step_by(2).map(|k| *k).collect();

        // This line is very important and could cause trouble in terms of the generic functions 
        let y_even: C = fft_ct::<I, C, F>(&x_even);
        let y_odd: C = fft_ct::<I, C, F>(&x_odd);

        let mut y = C::from_iter(std::iter::repeat(Complex::<F>::new(zero, zero)).take(N)); // preallocate memory
        // Need to get rid of the indexs here as they can panic!
        for j in 0..(N/2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + N/2] = y_even[j] - tmp; 
            w *= wn;
        }
        y
    }
}

impl<C> Fft for C
where 
    Self: IntoIterator + FromIterator<<Self as IntoIterator>::Item> + IndexMut<usize, Output = <Self as IntoIterator>::Item>,
    Self: Deref<Target = [<Self as IntoIterator>::Item]> + Clone,
    <Self as IntoIterator>::Item: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<<Self as IntoIterator>::Item>
{}



pub fn fft_pf<U, T>(x: &Array1<U>) -> Array1<Complex<T>> 
where 
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>,
{
    std::unimplemented!();
}

pub fn fft_bluestein<U, T>(x: &Array1<U>) -> Array1<Complex<T>> 
where 
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>,
{
    std::unimplemented!();
}

pub fn ifft_ct<T: FloatVal>(x: &Array1<Complex<T>>) -> Array1<T> 
{
    std::unimplemented!();
}


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
    #[allow(unused_imports)]
    #[allow(dead_code)]
    #[allow(unused_variables)]
    use super::*;
    use crate::io::{read_json, Data};
    use crate::test_utils as test;
    use ndarray::prelude::*;

    #[test]
    fn test_fft_ct_trait() {
        let json_data = read_json("datasets/fft/fft/fft.json");
        let output: Vec<Complex<f64>> = match json_data.input_data {
            Data::<f64>::Array(input) => input.fft_ct(),
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
    }

    #[test]
    fn test_fft_ct_func() {
        let json_data = read_json("datasets/fft/fft/fft.json");
        let output: Vec<Complex<f64>> = match json_data.input_data {
            Data::<f64>::Array(input) => fft_ct(&input),
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
    }

    #[test]
    fn test_fft_ct_array_func() {
        let json_data = read_json("datasets/fft/fft/fft.json");
        let output: Array1<Complex<f64>> = match json_data.input_data {
            Data::<f64>::Array(input) => fft_ct::<Vec<f64>, Array1<Complex<f64>>, f64>(&input),
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
    }
    
    #[test]
    fn dtft_sine() {
        std::unimplemented!();
    }

    #[test]
    fn fft_cooley_tukey_sine() {
        std::unimplemented!();
    }

    #[test]
    fn fftfreq() {
        std::unimplemented!();
    }

    #[test]
    fn zero_pad() {
        std::unimplemented!();
    }

    #[test]
    fn fft_macro() {
        std::unimplemented!();
    }

    #[test]
    fn fftfreq_macro() {
        std::unimplemented!();
    }
}


