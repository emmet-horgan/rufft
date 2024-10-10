use num_traits::float::Float;
use ndarray::prelude::*;
//use num_traits::cast::FromPrimitive;

pub fn sine<T>(f: T, fs: T, t: T) -> Array1<T>
where T: Float
{
    let pi: T = T::from(std::f64::consts::PI).unwrap();
    let len: usize = (t * fs).floor().to_usize().unwrap(); // Round down and cast to usize
    let t_step = T::from(1.0).unwrap() / fs;

    let mut signal: Array1<T> = Array1::zeros(len);
    for i in 0..len {
        let t: T = T::from(i).unwrap() * t_step;
        signal[i] = (T::from(2.0).unwrap() * pi * fs * t).sin();
    }
    return signal;
}

pub fn sinc() {
    std::unimplemented!();
}

pub fn triangular() {
    std::unimplemented!();
}

pub fn square() {
    std::unimplemented!();
}

pub fn sawtooth() {
    std::unimplemented!();
}

pub fn pulse() {
    std::unimplemented!();
}

pub fn multi_tone() {
    std::unimplemented!();
}

