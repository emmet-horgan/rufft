#![allow(non_snake_case)]
use ndarray::prelude::*;
use num::Zero;
use num_complex::Complex;
use crate::traits::{FloatVal, SigVal};
use num_traits::Num;
use num_traits::cast::AsPrimitive;
use std::ops::DivAssign;
use std::iter::{ExactSizeIterator, Iterator};

struct DftIter<I, T>
where
    I: ExactSizeIterator,
    I::Item: SigVal<T>,
    T: FloatVal
{
    iter: I,
    cur: usize,
    length: usize,
    _marker: std::marker::PhantomData<T>
}

impl<I, T> DftIter<I, T> 
where 
    I: ExactSizeIterator,
    I::Item: SigVal<T>,
    T: FloatVal
{
    fn new(iter: I) -> Self {
        let length = iter.len();
        DftIter {
            iter,
            cur: 0,
            length,
            _marker: std::marker::PhantomData
        }
    }
}

impl<I, T> Iterator for DftIter<I, T>
where
    I: ExactSizeIterator,
    I::Item: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>
{
    type Item = Complex<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.length - 1 {
            return None;
        }

        let mut sum = Complex::<T>::zero();
        let N = self.length;
        let N_t: T = N.as_();
        let twopi: T = (2).as_() * T::PI();
        let zero = T::zero();
        let k = self.cur;
        let mut n: usize = 0;
        for x in self.iter.by_ref() {
            let phase: Complex<T> = Complex::<T>::new(zero, -(twopi * n.as_() * k.as_()) / N_t);
            sum += Complex::<T>::new(x.as_(), zero) * phase.exp();
            n += 1
        }
        self.cur += 1;
        Some(sum)
    }
}

impl<I, T> ExactSizeIterator for DftIter<I, T>
where
    I: ExactSizeIterator,
    I::Item: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>
{
    fn len(&self) -> usize {
        self.length
    }
}

pub fn dft_func<T: FloatVal, I: ExactSizeIterator>(x: I) -> DftIter<I, T>
where
    I: ExactSizeIterator,
    I::Item: SigVal<T>,
    usize: AsPrimitive<T>
{
    DftIter::new(x)
}

pub fn dft<U, T>(x: &Array1<U>) -> Array1<Complex<T>>
where
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>
{

    let N = x.len();
    let N_t: T = N.as_();
    let twopi: T = (2).as_() * T::PI();
    let zero: T = T::zero();
    
    let mut out = Array1::<Complex<T>>::zeros(N); // preallocate memory

    for k in 0..N {
        let mut sum: Complex<T> = Complex::<T>::new(zero, zero);
        let k_t: T = k.as_();
        for n in 0..N {
            let n_t: T = n.as_();
            let phase: Complex<T> = Complex::<T>::new(zero, -(twopi * n_t * k_t) / N_t);
            let arg: Complex<T> = Complex::<T>::new(x[n].as_(), zero) * phase.exp();
            
            sum += arg;
        }
        out[k] = sum;
    }
    
    out
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

pub fn fft_ct<U, T>(x: &Array1<U>) -> Array1<Complex<T>> 
where 
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>,
{
    let N = x.len();
    let N_t: T = N.as_();
    let zero = T::zero();
    let one = T::one();
    let twopi: T = (2).as_() * T::PI();

    if N == 1 {
        array![Complex::<T>::new(x[0].as_(), zero)]
    }
    else {
        let wn = Complex::<T>::new(zero, -twopi / N_t);
        let wn = wn.exp();
        let mut w = Complex::<T>::new(one, zero);

        let x_even: Array1<U> = x.slice(s![0..;2]).to_owned();
        let x_odd: Array1<U> = x.slice(s![1..;2]).to_owned();
        
        let y_even = fft_ct(&x_even);
        let y_odd = fft_ct(&x_odd);

        let mut y = Array1::<Complex<T>>::zeros(N); // preallocate memory
        for j in 0..(N/2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + N/2] = y_even[j] - tmp; 
            w *= wn;
        }
        y
    }
}

pub fn fft_pf<U, T>(x: &Array1<U>) -> Array1<Complex<T>> 
where 
    U: SigVal<T>,
    T: FloatVal,
    usize: AsPrimitive<T>,
{
    std::unimplemented!();
}

pub fn czt<U, T>(x: &Array1<U>) -> Array1<Complex<T>> 
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

//pub fn pad<T: Num + Copy>(x: &Array1<T>, val: T, len: usize) -> Result<Array1<T>, ndarray::ShapeError> {
//    //let _: Array1<T> = x.iter().map(|x| x).into()
//    //let tmp = x.clone();
//    x.clone().into_shape((1, len))
//}

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

    use crate::fft;
    use crate::io;
    use ndarray::prelude::*;
    use num_complex::Complex;

    #[test]
    fn dft_sine() {
        let json_data = io::read_json("datasets/fft/fft/fft.json");
        let output: Array1<Complex<f64>>;
        let epsilon = 1E-10;
        let mut max_mag_error: f64 = 0.0;
        let mut max_phase_error: f64 = 0.0;
        match json_data.input_data {
            io::Data::<f64>::Array(input) => {
                let input: Array1<f64> = Array1::from_vec(input);
                output = fft!(&input);
            }
            _ => {panic!()}
        }
        match json_data.output_data {
            io::Data::ComplexVals { mag, phase} => {
                for i in 0..mag.len() {
                    let mag_calc = output[i].norm();
                    let phase_calc = output[i].arg();
                    let percentage_mag: f64;
                    let percentage_phase: f64;
                    if (mag[i] == f64::from(0.0)) || (mag_calc == f64::from(0.0)) {
                        percentage_mag = (mag[i] - mag_calc).abs();
                    } 
                    else {
                        percentage_mag = (mag[i] - mag_calc).abs() / mag[i];
                    }
                    if (phase[i] == f64::from(0.0)) || (phase_calc == f64::from(0.0)) {
                        percentage_phase = (phase[i] - phase_calc).abs();
                    } 
                    else {
                        percentage_phase = (phase[i] - phase_calc).abs() / phase[i];
                    }
                    if percentage_mag > max_mag_error {
                        max_mag_error = percentage_mag;
                    }
                    if percentage_phase > max_phase_error {
                        max_phase_error = percentage_phase;
                    }
                    if !((percentage_mag < epsilon) && (percentage_phase < epsilon)) {
                    assert!(false)
                    }
                    assert!((percentage_mag < epsilon) && (percentage_phase < epsilon));
                }
                println!("Maximum % magnitude error: {}", max_mag_error);
                println!("Maximum % phase error: {}", max_phase_error);
            }
            _ => {panic!()}
        }
    }

    #[test]
    fn dft_sine_func() {
        let json_data = io::read_json("datasets/fft/fft/fft.json");
        let mut output: Array1<Complex<f64>>;
        let epsilon = 1E-10;
        let mut max_mag_error: f64 = 0.0;
        let mut max_phase_error: f64 = 0.0;
        match json_data.input_data {
            io::Data::<f64>::Array(input) => {
                let input: Array1<f64> = Array1::from_vec(input);
                output = Array1::zeros()
                let result = fft::dft_func(input.into_iter());
                for x in result {
                    println!("{}", x);
                }
            }
            _ => {panic!()}
        }
        //match json_data.output_data {
        //    io::Data::ComplexVals { mag, phase} => {
        //        for i in 0..mag.len() {
        //            let mag_calc = output[i].norm();
        //            let phase_calc = output[i].arg();
        //            let percentage_mag: f64;
        //            let percentage_phase: f64;
        //            if (mag[i] == f64::from(0.0)) || (mag_calc == f64::from(0.0)) {
        //                percentage_mag = (mag[i] - mag_calc).abs();
        //            } 
        //            else {
        //                percentage_mag = (mag[i] - mag_calc).abs() / mag[i];
        //            }
        //            if (phase[i] == f64::from(0.0)) || (phase_calc == f64::from(0.0)) {
        //                percentage_phase = (phase[i] - phase_calc).abs();
        //            } 
        //            else {
        //                percentage_phase = (phase[i] - phase_calc).abs() / phase[i];
        //            }
        //            if percentage_mag > max_mag_error {
        //                max_mag_error = percentage_mag;
        //            }
        //            if percentage_phase > max_phase_error {
        //                max_phase_error = percentage_phase;
        //            }
        //            if !((percentage_mag < epsilon) && (percentage_phase < epsilon)) {
        //            assert!(false)
        //            }
        //            assert!((percentage_mag < epsilon) && (percentage_phase < epsilon));
        //        }
        //        println!("Maximum % magnitude error: {}", max_mag_error);
        //        println!("Maximum % phase error: {}", max_phase_error);
        //    }
        //   _ => {panic!()}
        //}
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


