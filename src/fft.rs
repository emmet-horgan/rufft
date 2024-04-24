#![allow(non_snake_case)]
use ndarray::{prelude::*};
use num_complex::{Complex};
use crate::traits::SigVal;
use num_traits::float;
use num_traits::cast::{AsPrimitive, FromPrimitive};

pub fn dft<T>(x: &Array1<T>) -> Array1<Complex<T>>
where
    T: SigVal,
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
            let arg: Complex<T> = Complex::<T>::new(x[n], zero) * phase.exp();
            
            sum += arg;
        }
        out[k] = sum;
    }
    
    return out;
}

pub fn fftfreq<T>(n: usize, p: T) -> Array1<T> 
where 
    T: float::Float + FromPrimitive + ndarray::ScalarOperand + std::ops::DivAssign,
{
    let zero = T::from_f32(0.0).unwrap();
    let one = T::from_f32(1.0).unwrap();
    let two = T::from_f32(2.0).unwrap();
    let n_t = T::from_usize(n).unwrap();
    

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
    arr /= p * n_t;
    return arr;
}

pub fn dtft<'a, T>(x: &'a Array1<T>) -> impl Fn(&Array1<T>) -> Array1<Complex<T>> + 'a
where 
    T: float::Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
{
    |samples: &Array1<T>| -> Array1<Complex<T>> {
        let zero = T::from_f32(0.0).unwrap();
        let mut y: Array1<Complex<T>> = Array1::<Complex<T>>::zeros(samples.len());

        for (index, &w) in samples.iter().enumerate() {
            let mut sum: Complex<T> = Complex::<T>::new(zero, zero);
            for n in 0..x.len() {
                let n_t = T::from_usize(n).unwrap();
                let phase: Complex<T> = Complex::<T>::new(zero, -n_t * w);
                
                let arg: Complex<T> = Complex::<T>::new(x[n], zero) * phase.exp();
                
                sum += arg;
            }
            y[index] = sum;
        }
        y
    }
}

pub fn fft_ct<T>(x: &Array1<T>) -> Array1<Complex<T>> 
where 
    T: float::Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
{
    let N = x.len();
    let N_t = T::from_usize(N).unwrap();
    let zero = T::from_f64(0.0).unwrap();
    let one = T::from_f64(1.0).unwrap();
    let twopi = T::from_f64(std::f64::consts::PI * 2.0).unwrap();

    if N == 1 {
        return array![Complex::<T>::new(x[0], zero)];
    }
    else {
        let wn = Complex::<T>::new(zero, -twopi / N_t);
        let wn = wn.exp();
        let mut w = Complex::<T>::new(one, zero);

        let x_even: Array1<T> = x.slice(s![0..;2]).to_owned();
        let x_odd: Array1<T> = x.slice(s![1..;2]).to_owned();
        
        let y_even = fft_ct(&x_even);
        let y_odd = fft_ct(&x_odd);

        let mut y = Array1::<Complex<T>>::zeros(N); // preallocate memory
        for j in 0..(N/2) {
            let tmp = w * y_odd[j];
            y[j] = y_even[j] + tmp;
            y[j + N/2] = y_even[j] - tmp; 
            w *= wn;
        }
        return y;
    }
}

pub fn fft_pf<T>(x: &Array1<T>) -> Array1<Complex<T>> 
where 
    T: float::Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
{
    std::unimplemented!();
}

pub fn ifft_ct<T>(x: &Array1<Complex<T>>) -> Array1<T> 
where 
    T: float::Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
{
    std::unimplemented!();
}

pub fn zero_pad<T>(x: &Array1<T>, ) -> Option<Array1<T>> 
where 
    T: float::Float + FromPrimitive
{
    let zero = T::from_f64(0.0).unwrap();
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
                fft::dft(&$x)
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


