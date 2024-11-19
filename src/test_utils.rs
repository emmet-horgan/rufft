use num_traits::{Float, FloatConst, NumAssignOps, AsPrimitive};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use serde_json;
use std::path::{PathBuf};
use std::env;
use std::fs;
use walkdir::{DirEntry, WalkDir};

struct PathManage {
    _root: PathBuf
}

impl PathManage {

    fn new(root: &str) -> PathManage {
        PathManage{_root: PathManage::find_root(root).unwrap()}
    }

    fn split(path: &PathBuf) -> Vec<&str> {
        path.components().map(|c| c.as_os_str().to_str().unwrap()).collect::<Vec<&str>>()
    }

    fn is_hidden(entry: &DirEntry) -> bool {
        entry.file_name()
             .to_str()
             .map(|s| s.starts_with("."))
             .unwrap_or(false)
    }

    fn path(&self, path: &str) -> Option<PathBuf> {
        let path_pathbuf = PathBuf::from(path);
        let path_parts = Self::split(&path_pathbuf);
        let path_parts: Vec<&str> = path_parts.into_iter().filter(|&part| part != "." && part != ".." && !part.is_empty()).collect();
        let length = path_parts.len();

        for entry in WalkDir::new(self._root
                                                    .as_os_str()
                                                    .to_str().unwrap())
                                                    .into_iter()
                                                    .filter_map(|e| e.ok()) 
        {
            let entry_path: PathBuf = entry.path().to_path_buf();
            let entry_parts = Self::split(&entry_path);
            if (entry_parts.len() as i32) - (length as i32) < 1 { // Avoid neg index and cast to i32 to avoid overflows
                continue;
            }
            let semipath = entry_parts[entry_parts.len() - length ..].to_vec();
            if path_parts.iter().all(|item| semipath.contains(item)) {
                return Some(path_pathbuf);
            }

        }

        None
    }

    fn find_root(root_name: &str) -> Result<PathBuf, std::io::Error> {
        let current_dir = env::current_dir()?;
        let mut path = current_dir.components().collect::<Vec<_>>();
        path.reverse();

        let mut dir = PathBuf::new();
        dir.push(".");

        for component in path {
            if let Some(name) = component.as_os_str().to_str() {
                if name != root_name {
                    dir.push("..");
                } else {
                    return Ok(dir);
                }
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Root directory not found",
        ))
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub enum Data<T> {
    FftFreqVals {
        n: T,
        d: T
    },

    ComplexVals {
        mag: Vec<T>,
        phase: Vec<T>
    },

    Array(Vec<T>),

    SineFreqVals {
        fsine: T,
        fsample: T,
        duration: T
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Json<F: Float + FloatConst + NumAssignOps + 'static> {
    pub input_data: Data<F>,
    pub output_data: Data<F>, 
    pub function: String,
    pub path: String
}

pub fn read_json<'a, F: Float + FloatConst + NumAssignOps + 'static + for<'de> Deserialize<'de>>(lib_path: &str) -> Json<F> {
    let path = PathManage::new("rufft");
    let json_path = path.path(lib_path).unwrap();
    let file = fs::File::open(json_path).unwrap();
    let data: Json<F> = serde_json::from_reader(file).unwrap();
    return data;
}


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
        let json_data = crate::test_utils::read_json($path);
        let output: $C = match json_data.input_data {
            crate::test_utils::Data::<$F>::Array(input) => $func::<$F, $I, $C>(&(input.into())),
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.output_data {
            crate::test_utils::Data::ComplexVals { mag, phase } => {
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
        let json_data = crate::test_utils::read_json($path);
        let output: $C = match json_data.output_data {
            crate::test_utils::Data::ComplexVals { mag, phase } => {
                let input = mag.iter().zip(phase.iter()).map(|(m, p)| num_complex::Complex::from_polar(*m, *p)).collect::<Vec<_>>();
                $func::<$F, $I, $C>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.input_data {
            crate::test_utils::Data::<$F>::Array(reference) => {
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
        let json_data = crate::test_utils::read_json($path);
        let output: $I = match json_data.input_data {
            crate::test_utils::Data::ComplexVals { mag, phase } => {
                let input = mag.iter().zip(phase.iter()).map(|(m, p)| num_complex::Complex::from_polar(*m, *p)).collect::<$I>();
                $func::<$F, $I>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.output_data {
            crate::test_utils::Data::ComplexVals { mag, phase } => {
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
        let json_data = crate::test_utils::read_json($path);
        let output: $I = match json_data.output_data {
            crate::test_utils::Data::ComplexVals { mag, phase } => {
                let input = mag.iter().zip(phase.iter()).map(|(m, p)| num_complex::Complex::from_polar(*m, *p)).collect::<$I>();
                $func::<$F, $I, $C>(&(input.into()))
            },
            _ => panic!("Read the input data incorrectly")
        };
        match json_data.input_data {
            crate::test_utils::Data::ComplexVals { mag, phase } => {
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