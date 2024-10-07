#![allow(non_snake_case)]
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
            x.into_iter().enumerate().map(|(n, j)| {
                let phase = Complex::<F>::new(F::zero(), -n.as_() * w);
                Complex::<F>::new(*j, F::zero()) * phase.exp()
            }).sum()
        }).collect()
    }
}


pub fn idft<I, C, F>(x: &I) -> C
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
    let N = x.into_iter().len();
    let zero = F::zero();
    let twopi = F::TAU();
    x.into_iter().enumerate().map(|(k, _)|{
        x.into_iter().enumerate().map(|(n, j)| {
            let phase = Complex::<F>::new(zero, (twopi * n.as_() * k.as_()) / N.as_());
            Complex::<F>::new(*j, zero) * phase.exp()
        }).sum::<Complex<F>>() * Complex::<F>::new(F::one() / N.as_(), zero)
    }).collect()
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
    let N = x.into_iter().len();
    let zero = F::zero();
    let twopi = F::TAU();
    x.into_iter().enumerate().map(|(k, _)|{
        x.into_iter().enumerate().map(|(n, j)| {
            let phase = Complex::<F>::new(zero, -(twopi * n.as_() * k.as_()) / N.as_());
            Complex::<F>::new(*j, zero) * phase.exp()
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

//pub fn fftfreq(n: usize, d: f64) -> Array1<f64> {
//where 
//    // Bound F to float types
//    F: Float + FloatConst + NumAssign + 'static,
//    C: FromIterator<F> + Clone,
//{
//    let zero = F::zero();
//    let one = F::one();
//    let two = F::TAU();
//    let n_t: F = n.as_();
//    
//    let part0 :Array1<T>;
//    let part1 :Array1<T>;
//    if n % 2 == 0 {
//        part0  = Array1::<T>::range(zero, n_t / two, one);
//        part1  = Array1::<T>::range(-n_t / two, -zero, one);
//        
//    }
//    else {
//        part0 = Array1::<T>::range(zero, (n_t - one) / two, one);
//        part1 = Array1::<T>::range(-(n_t - one) / two, -zero, one);
//    }
//
//    let mut arr = ndarray::concatenate![Axis(0), part0, part1];
//    
//    arr /= p.as_() * n_t;
//    return arr;
//}

//pub fn zero_pad<T: Num + Copy>(x: &Array1<T>) -> Option<Array1<T>> 
//{
//    let zero = T::zero();
//    let N = x.len();
//
//    // Add a check for N = 2^{m}, zero pad if not. Only executed once regardsless of recursion.
//    if (N != 0) && (N & (N - 1)) != 0 {
//        let mut power = 1;
//        while power < N {
//            power <<= 1;
//        }
//        let mut y = Array1::<T>::zeros(power);
//        for i in 0..N {
//            y[i] = x[i];
//        }
//        for i in N..power {
//            y[i] = zero;
//        }
//        return Some(y);
//    }
//    else {
//        return None;
//    }
//} 

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
    use crate::traits::Fft; // For methods
    use crate::io::{read_json, Data};
    use crate::test_utils as test;
    use ndarray::prelude::*;

    const ATOL_F64: f64 = 1e-12;
    const RTOL_F64: f64 = 1e-9;

    // Really loose tolerances for f32 because we're checking complex numbers
    // which is more difficult, especially near zero where the phase can suddenly
    // jump by π for a small change in the real or imaginary part.
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
                //assert!(test::nearly_equal(dft_comp.norm(), fft_comp.norm()), 
                //    "complex[{:?} != {:?}] {} != {}", dft_comp, fft_comp, dft_comp.norm(), fft_comp.norm());
                //assert!(test::nearly_equal(dft_comp.arg(), fft_comp.arg()), 
                //    "complex[{:?} != {:?}] {} != {}", dft_comp, fft_comp, dft_comp.arg(), fft_comp.arg());
            }
        }
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
        //test_dft_func!(Vec<f64>, Vec<Complex<f64>>, f64);
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
}


