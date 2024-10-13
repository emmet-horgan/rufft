use num_traits::{Float, AsPrimitive};
use num_complex::Complex;

pub fn nearly_equal<F: Float + 'static>(a: F, b: F, rtol: F, atol: F) -> bool 
{
    (a - b).abs() <= atol + rtol * b.abs()
}


pub fn nearly_equal_complex<F: Float + 'static>(a: Complex<F>, b: Complex<F>, rtol: F, atol: F) -> bool 
where 
    f64: AsPrimitive<F>
{
    let mag_a = a.norm();
    let mag_b = b.norm();
    
    // Compare magnitudes first
    if !nearly_equal(mag_a, mag_b, rtol, atol) {
        return false;
    }

    // If any value is close to zero, it flip the angle's sign
    if (a.re < atol && b.re < atol) || (a.im < atol && b.im < atol) {
        return true;
    }
    // Compare the phase only if magnitudes are sufficiently large
    nearly_equal(a.arg(), b.arg(), rtol, atol)
}

macro_rules! test_fourier_transform {
    ($func:ident, $path:literal, $F:ty, $I:ty, $C:ty, $rtol:expr, $atol:expr) => {
        let json_data = crate::io::read_json($path);
        let output: $C = match json_data.input_data {
            crate::io::Data::<$F>::Array(input) => $func::<$F, $I, $C>(&(input.into())),
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.output_data {
            crate::io::Data::ComplexVals { mag, phase } => {
                for i in 0..mag.len() {
                    let reference = num_complex::Complex::from_polar(mag[i], phase[i]);
                    assert!(crate::test_utils::nearly_equal_complex(output[i], reference, $rtol, $atol), 
                        "{} => {} != {}", i, output[i], reference);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    };
}

macro_rules! test_ifourier_transform {
    ($func:ident, $path:literal, $F:ty, $I:ty, $C:ty, $rtol:expr, $atol:expr) => {
        let json_data = crate::io::read_json($path);
        let output: $C = match json_data.output_data {
            crate::io::Data::ComplexVals { mag, phase } => {
                let input = mag.iter().zip(phase.iter()).map(|(m, p)| num_complex::Complex::from_polar(*m, *p)).collect::<Vec<_>>();
                $func::<$F, $I, $C>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.input_data {
            crate::io::Data::<$F>::Array(reference) => {
                for i in 0..reference.len() {
                    assert!(crate::test_utils::nearly_equal(output[i], reference[i], $rtol, $atol), 
                        "{} => {} != {}", i, output[i], reference[i]);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    };
}

macro_rules! test_complex_fourier_transform {
    ($func:ident, $path:literal, $F:ty, $I:ty, $rtol:expr, $atol:expr) => {
        let json_data = crate::io::read_json($path);
        let output: $I = match json_data.input_data {
            crate::io::Data::ComplexVals { mag, phase } => {
                let input = mag.iter().zip(phase.iter()).map(|(m, p)| num_complex::Complex::from_polar(*m, *p)).collect::<$I>();
                $func::<$F, $I>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.output_data {
            crate::io::Data::ComplexVals { mag, phase } => {
                for i in 0..mag.len() {
                    let reference = num_complex::Complex::from_polar(mag[i], phase[i]);
                    assert!(crate::test_utils::nearly_equal_complex(output[i], reference, $rtol, $atol), 
                        "{} => {} != {}", i, output[i], reference);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    };
}

macro_rules! test_complex_ifourier_transform {
    ($func:ident, $path:literal, $F:ty, $I:ty, $C:ty, $rtol:expr, $atol:expr) => {
        let json_data = crate::io::read_json($path);
        let output: $I = match json_data.output_data {
            crate::io::Data::ComplexVals { mag, phase } => {
                let input = mag.iter().zip(phase.iter()).map(|(m, p)| num_complex::Complex::from_polar(*m, *p)).collect::<$I>();
                $func::<$F, $I, $C>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.input_data {
            crate::io::Data::ComplexVals { mag, phase } => {
                for i in 0..mag.len() {
                    let reference = num_complex::Complex::from_polar(mag[i], phase[i]);
                    assert!(crate::test_utils::nearly_equal_complex(output[i], reference, $rtol, $atol), 
                        "{} => {} != {}", i, output[i], reference);
                }
            }
            _ => panic!("Read the output data incorrectly")
        }
    };
}


macro_rules! test_fft {
    ($F:ty,$I:ty, $C:ty, $rtol:expr, $atol:expr) => {
        crate::test_utils::test_fourier_transform!(fft, "datasets/fft/fft/fft.json", $F, $I, $C, $rtol, $atol);
    };
}

macro_rules! test_dft {
    ($F:ty,$I:ty, $C:ty, $rtol:expr, $atol:expr) => {
        crate::test_utils::test_fourier_transform!(dft, "datasets/fft/fft/fft.json", $F, $I, $C, $rtol, $atol);
    };
}

macro_rules! test_complex_fft {
    ($F:ty,$I:ty, $rtol:expr, $atol:expr) => {
        crate::test_utils::test_complex_fourier_transform!(fft, "datasets/fft/complex_fft/complex_fft.json", $F, $I, $rtol, $atol);
    };
}

macro_rules! test_complex_dft {
    ($F:ty,$I:ty, $rtol:expr, $atol:expr) => {
        crate::test_utils::test_complex_fourier_transform!(dft, "datasets/fft/complex_fft/complex_fft.json", $F, $I, $rtol, $atol);
    };
}

macro_rules! test_complex_ifft {
    ($F:ty,$I:ty, $rtol:expr, $atol:expr) => {
        crate::test_utils::test_complex_ifourier_transform!(ifft, "datasets/fft/complex_fft/complex_fft.json", $F, $I, $I, $rtol, $atol);
    };
}

macro_rules! test_idft {
    ($F:ty,$I:ty, $C:ty, $rtol:expr, $atol:expr) => {
        crate::test_utils::test_ifourier_transform!(idft, "datasets/fft/fft/fft.json", $F, $I, $C, $rtol, $atol);
    };
}


pub(crate) use test_fft;
pub(crate) use test_dft;
pub(crate) use test_complex_fft;
pub(crate) use test_complex_dft;
pub(crate) use test_complex_ifft;
pub(crate) use test_idft;

pub(crate) use test_fourier_transform;
pub(crate) use test_complex_fourier_transform;
pub(crate) use test_complex_ifourier_transform;
pub(crate) use test_ifourier_transform;