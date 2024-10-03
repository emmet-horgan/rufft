#![allow(non_snake_case)]
use ndarray::prelude::*;
use num::{zero, One, Zero};
use num_complex::{Complex, ComplexFloat};
use crate::traits::{FloatVal, SigVal};
use num_traits::{Float, FloatConst, Num, NumAssign, AsPrimitive};
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

pub trait Fft: IntoIterator + Deref<Target = [<Self as IntoIterator>::Item]> + Sized + Clone + FromIterator<<Self as IntoIterator>::Item> + IndexMut<usize, Output = <Self as IntoIterator>::Item>
where 
    <Self as IntoIterator>::Item: ComplexFloat + HasInnerFloat,// + Deref<Target = <Self as IntoIterator>::Item>,
    usize: AsPrimitive<<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat>,
    //<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat: From<usize>,
    <Self as IntoIterator>::Item: TyEq<Type = Complex<<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat>>,
    //<Self as Iterator>::Item = Complex<<<Self as Iterator>::Item as HasInnerFloat>::InnerFloat>,
    //Complex<<<Self as Iterator>::Item as HasInnerFloat>::InnerFloat>: Mul<<Self as Iterator>::Item>
{
    //type Type;

    fn fft(&self) -> Self {
        let N: usize = self.len();
        let N_calc: <<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat = N.as_();
        
        //let zero = <Self as Iterator>::Item::zero();
        //let one = <Self as Iterator>::Item::one();
        let zero = <<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat>::zero();
        let one = <<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat>::one();
        let twopi = <<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat as FloatConst>::TAU();

        if N == 1 {
            self.clone()
            //self.into_iter().map(|x| Complex::<<<Self as Iterator>::Item as HasInnerFloat>::InnerFloat>::new(x, zero)).collect()
        } else {
            let wn = Complex::<<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat>::new(zero, -twopi / N_calc );
            //let wn: <Self as Iterator>::Item = wn.exp().into();
            let wn = wn.exp();
            //let mut w: <Self as Iterator>::Item = Complex::<<<Self as Iterator>::Item as HasInnerFloat>::InnerFloat>::new(one, zero).into();
            let mut w = Complex::<<<Self as IntoIterator>::Item as HasInnerFloat>::InnerFloat>::new(one, zero);
            //let wn = Complex::<U>::new(zero, -twopi / N.as_());
            //let wn = wn.exp();
            //let mut w = Complex::<U>::new(one, zero);
            
            let x_even: Self = self.clone().into_iter().step_by(2).map(|k| k).collect();
            let x_odd: Self =  self.clone().into_iter().skip(1).step_by(2).map(|k| k).collect();
            //let x_even: Array1<T> = x.into_iter().step_by(2).map(|k| *k).collect();
            //let x_odd: Array1<T> = x.into_iter().skip(1).step_by(2).map(|k| *k).collect();
            
            // This line is very important and could cause trouble in terms of the generic functions 
            // created. Perhaps better to use an output type of C and make C indexable ?
            let y_even = x_even.fft();
            let y_odd = x_odd.fft();
            //let y_even: Array1<_> = fft_ct(x_even.as_slice().unwrap());
            //let y_odd: Array1<_> = fft_ct(x_odd.as_slice().unwrap());
            
            let mut y = self.clone(); // preallocate memory
            for j in 0..(N/2) {
                let tmp = <Self as IntoIterator>::Item::from(w) * y_odd[j];
                y[j] = y_even[j] + tmp;
                y[j + N/2] = y_even[j] - tmp; 
                w *= wn;
            }
            y

            //let mut y = Array1::<Complex<U>>::zeros(N); // preallocate memory
            //for j in 0..(N/2) {
            //    let tmp = w * y_odd[j];
            //    y[j] = y_even[j] + tmp;
            //    y[j + N/2] = y_even[j] - tmp; 
            //    w *= wn;
            //}
            //y.iter().map(|k| *k).collect();
            //self.clone()
        }
    }
    //fn ifft(&self) -> Self;
}

pub fn generic_fft<F, I, C>(x: &[F]) -> C
where 
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>,
    C: IntoIterator<Item = Complex<F>> + FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>,
    I: IntoIterator<Item = F> + FromIterator<F> + Deref<Target = [F]>,
{
    let N = x.len();
    let zero = F::zero();
    let one = F::one();
    let twopi = F::TAU();

    if N == 1 {
        return x.into_iter().map(|x| Complex::<F>::new(*x, zero)).collect();
    } else {
        let wn = Complex::<F>::new(zero, -twopi / N.as_());
        let wn = wn.exp();
        let mut w = Complex::<F>::new(one, zero);

        let x_even: I = x.into_iter().step_by(2).map(|k| *k).collect();
        let x_odd: I = x.into_iter().skip(1).step_by(2).map(|k| *k).collect();

        // This line is very important and could cause trouble in terms of the generic functions 
        // created. Perhaps better to use an output type of C and make C indexable ?
        let y_even: C = generic_fft::<F, I, C>(&x_even);
        let y_odd: C = generic_fft::<F, I, C>(&x_odd);

        let mut y = C::from_iter(std::iter::repeat(Complex::<F>::new(zero, zero)).take(N)); // preallocate memory
        for j in 0..(N/2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + N/2] = y_even[j] - tmp; 
            w *= wn;
        }
        y
    }
}

impl Fft for Vec<Complex<f64>> {}

pub fn fft_ct<T, U, C>(x: &[T]) -> C 
where 
    T: SigVal<U>,
    U: FloatVal,
    usize: AsPrimitive<U>,
    C: FromIterator<Complex<U>>
{
    let N = x.len();
    let zero = U::zero();
    let one = U::one();
    let twopi: U = (2).as_() * U::PI();

    if N == 1 {
        x.into_iter().map(|x| Complex::<U>::new(x.clone().as_(), zero)).collect()
    } else {
        let wn = Complex::<U>::new(zero, -twopi / N.as_());
        let wn = wn.exp();
        let mut w = Complex::<U>::new(one, zero);

        let x_even: Array1<T> = x.into_iter().step_by(2).map(|k| *k).collect();
        let x_odd: Array1<T> = x.into_iter().skip(1).step_by(2).map(|k| *k).collect();
        
        // This line is very important and could cause trouble in terms of the generic functions 
        // created. Perhaps better to use an output type of C and make C indexable ?
        let y_even: Array1<_> = fft_ct(x_even.as_slice().unwrap());
        let y_odd: Array1<_> = fft_ct(x_odd.as_slice().unwrap());
        
        let mut y = Array1::<Complex<U>>::zeros(N); // preallocate memory
        for j in 0..(N/2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + N/2] = y_even[j] - tmp; 
            w *= wn;
        }
        y.iter().map(|k| *k).collect()
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
    use super::*;

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
                output = fft::fft_ct(input.as_slice().unwrap());
                println!("len(output) for slice = {}", output.len());
                println!("output[0] = {}", output[0]);
                println!("output[1] = {}", output[1]);
                println!("output[2] = {}", output[2]);
                println!("output[3] = {}", output[3]);
            }
            _ => {panic!()}
        }
        match json_data.output_data {
            io::Data::ComplexVals { mag, phase} => {
                let mut err_mag_acc: f64 = 0.0;
                let mut err_phase_acc: f64 = 0.0;
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
                    err_mag_acc += percentage_mag;
                    err_phase_acc += percentage_phase;
                    //if !((percentage_mag < epsilon) && (percentage_phase < epsilon)) {
                    //    println!("percentage_mag = {}", percentage_mag);
                    //    println!("percentage_phase = {}", percentage_phase);
                    //    assert!(false)
                    //}
                    //assert!((percentage_mag < epsilon) && (percentage_phase < epsilon));
                }
                println!("Maximum % magnitude error: {}", max_mag_error);
                println!("Maximum % phase error: {}", max_phase_error);
                println!("Averave % magnitude error: {}", err_mag_acc / mag.len() as f64);
                println!("Averave % phase error: {}", err_phase_acc / mag.len() as f64);
            }
            _ => {panic!()}
        }
    }

    #[test]
    fn fft_trait() {
        let json_data = io::read_json("datasets/fft/fft/fft.json");
        let output: Vec<Complex<f64>>;
        let epsilon = 1E-10;
        let mut max_mag_error: f64 = 0.0;
        let mut max_phase_error: f64 = 0.0;
        match json_data.input_data {
            io::Data::<f64>::Array(input) => {
                //let input: Array1<f64> = Array1::from_vec(input);
                let input: Vec<Complex<f64>> = input.into_iter().map(|x| Complex::<f64>::new(x, 0.0)).collect();
                output = input.fft();
                //output = fft::fft_ct(input.as_slice().unwrap());
                println!("len(output) for slice = {}", output.len());
                println!("output[0] = {}", output[0]);
                println!("output[1] = {}", output[1]);
                println!("output[2] = {}", output[2]);
                println!("output[3] = {}", output[3]);
            }
            _ => {panic!()}
        }
        match json_data.output_data {
            io::Data::ComplexVals { mag, phase} => {
                let mut err_mag_acc: f64 = 0.0;
                let mut err_phase_acc: f64 = 0.0;
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
                    err_mag_acc += percentage_mag;
                    err_phase_acc += percentage_phase;
                    //if !((percentage_mag < epsilon) && (percentage_phase < epsilon)) {
                    //    println!("percentage_mag = {}", percentage_mag);
                    //    println!("percentage_phase = {}", percentage_phase);
                    //    assert!(false)
                    //}
                    //assert!((percentage_mag < epsilon) && (percentage_phase < epsilon));
                }
                println!("Maximum % magnitude error: {}", max_mag_error);
                println!("Maximum % phase error: {}", max_phase_error);
                println!("Averave % magnitude error: {}", err_mag_acc / mag.len() as f64);
                println!("Averave % phase error: {}", err_phase_acc / mag.len() as f64);
            }
            _ => {panic!()}
        }
    }

    #[test]
    fn generic_fft_test() {
        let json_data = io::read_json("datasets/fft/fft/fft.json");
        let output: Vec<Complex<f64>>;
        let epsilon = 1E-10;
        let mut max_mag_error: f64 = 0.0;
        let mut max_phase_error: f64 = 0.0;
        match json_data.input_data {
            io::Data::<f64>::Array(input) => {
                //let input: Array1<f64> = Array1::from_vec(input);
                //let input: Vec<Complex<f64>> = input.into_iter().map(|x| Complex::<f64>::new(x, 0.0)).collect();
                output = generic_fft::<f64, Vec<f64>, Vec<Complex<f64>>>(&input);
                //output = fft::fft_ct(input.as_slice().unwrap());
                println!("len(output) for slice = {}", output.len());
                println!("output[0] = {}", output[0]);
                println!("output[1] = {}", output[1]);
                println!("output[2] = {}", output[2]);
                println!("output[3] = {}", output[3]);
            }
            _ => {panic!()}
        }
        match json_data.output_data {
            io::Data::ComplexVals { mag, phase} => {
                let mut err_mag_acc: f64 = 0.0;
                let mut err_phase_acc: f64 = 0.0;
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
                    err_mag_acc += percentage_mag;
                    err_phase_acc += percentage_phase;
                    //if !((percentage_mag < epsilon) && (percentage_phase < epsilon)) {
                    //    println!("percentage_mag = {}", percentage_mag);
                    //    println!("percentage_phase = {}", percentage_phase);
                    //    assert!(false)
                    //}
                    //assert!((percentage_mag < epsilon) && (percentage_phase < epsilon));
                }
                println!("Maximum % magnitude error: {}", max_mag_error);
                println!("Maximum % phase error: {}", max_phase_error);
                println!("Averave % magnitude error: {}", err_mag_acc / mag.len() as f64);
                println!("Averave % phase error: {}", err_phase_acc / mag.len() as f64);
            }
            _ => {panic!()}
        }
    }

    #[test]
    fn dft_sine_func() {
        let json_data = io::read_json("datasets/fft/fft/fft.json");
        let output: Array1<Complex<f64>>;
        let epsilon = 1E-6;

        match json_data.input_data {
            io::Data::<f64>::Array(input) => {
                let input: Array1<f64> = Array1::from_vec(input);
                let tmp = input.as_slice().unwrap();
                output = fft::DftIter::new(tmp).collect();
            }
            _ => {panic!()}
        }
        match json_data.output_data {
            io::Data::ComplexVals { mag, phase} => {
                let mut err_mag_acc: f64 = 0.0;
                let mut err_phase_acc: f64 = 0.0;
                let mut max_pecent_mag_err: f64 = 0.0;
                let mut max_pecent_phase_err: f64 = 0.0;
                let mut max_mag_err: f64 = 0.0;
                let mut max_phase_err: f64 = 0.0;
                for (i, x) in output.into_iter().enumerate() {
                    let mag_calc = x.norm();
                    let phase_calc = x.arg();
                    let err_mag = (mag_calc - mag[i]).abs();
                    let err_phase = (phase_calc - phase[i]).abs();

                    let percentage_mag: f64 = if mag[i] != 0.0 {
                        err_mag / mag[i]
                    } else {
                        err_mag
                    };

                    let percentage_phase: f64 = if phase[i] != 0.0 {
                        err_phase / phase[i]
                    } else {
                        err_phase.abs()
                    };
                    
                    err_mag_acc += err_mag;
                    err_phase_acc += err_phase;
                    max_pecent_mag_err = if percentage_mag > max_pecent_mag_err {
                        percentage_mag
                    } else {
                        max_pecent_mag_err
                    };
                    max_pecent_phase_err = if percentage_phase > max_pecent_phase_err {
                        percentage_phase
                    } else {
                        max_pecent_phase_err
                    };

                    max_mag_err = if err_mag > max_mag_err {
                        err_mag
                    } else {
                        max_mag_err
                    };
                    max_phase_err = if err_phase > max_phase_err {
                        println!("phase biggest err index = {}", i);
                        println!("x = {}", x);
                        println!("phase(rust) = {}", x.arg());
                        println!("phase(python) = {}", phase[i]);
                        err_phase
                    } else {
                        max_phase_err
                    };
                }
                println!("Maximum % magnitude error: {}", max_pecent_mag_err);
                println!("Maximum % phase error: {}", max_pecent_phase_err);
                println!("Maximum magnitude error: {}", max_mag_err);
                println!("Maximum phase error: {}", max_phase_err);
                println!("Acc magnitude error: {}", err_mag_acc);
                println!("Acc phase error: {}", err_phase_acc);

                println!("Average % magnitude error: {}", err_mag_acc / mag.len() as f64);
                println!("Average % phase error: {}", err_phase_acc / mag.len() as f64);
            }
            _ => {panic!()}
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


