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
    let n = x.into_iter().len();
    let zero = F::zero();
    let twopi = F::TAU();
    x.into_iter().enumerate().map(|(i, _)|{
        x.into_iter().enumerate().map(|(j, &f)| {
            let phase = Complex::<F>::new(zero, (twopi * j.as_() * i.as_()) / n.as_());
            Complex::<F>::new(f, zero) * phase.exp()
        }).sum::<Complex<F>>() * Complex::<F>::new(F::one() / n.as_(), zero)
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
}


