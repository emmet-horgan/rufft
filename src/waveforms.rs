use thiserror::Error;
use num_traits::Float;
use core::fmt::Debug;
pub mod sine;


#[derive(Error, Debug)]
pub enum SamplingError<F: Float + Debug> {
    #[error("the sampling rate ({fs:?}) must be at least twice the 
        highest frequency in the signal ({f:?})")]
    ShannonError {
        f: F,
        fs: F
    },
    #[error("a sampling rate is required")]
    MissingSamplingFreq,
    #[error("a signal frequency is required")]
    MissingFreq
}