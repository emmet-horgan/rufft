use ndarray::{prelude::*};
use num_complex::{Complex};
use num_traits::float::Float;
use num_traits::cast::FromPrimitive;

pub fn dft<T>(x: &Array1<T>) -> Array1<Complex<T>>
where 
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
{

    let N = x.len();
    let N_t = T::from_usize(N).unwrap();
    let twopi: T = T::from_f64(std::f64::consts::PI * 2.0).unwrap();
    let zero = T::from_f32(0.0).unwrap();
    
    let mut out = Array1::<Complex<T>>::zeros(N); // preallocate memory

    for k in 0..N {
        let mut sum: Complex<T> = Complex::<T>::new(zero, zero);
        let k_t = T::from_usize(k).unwrap();
        for n in 0..N {
            let n_t = T::from_usize(n).unwrap();
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
    T: Float + FromPrimitive + ndarray::ScalarOperand + std::ops::DivAssign,
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
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
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
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign,
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

pub fn zero_pad<T>(x: &Array1<T>, ) -> Option<Array1<T>> 
where 
    T: Float + FromPrimitive
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
        match ft::zero_pad(&$x) {
            None => {
                ft::fft_ct(&$x)
            },
            Some(x) => {
                let tmp = ft::fft_ct(&x);
                tmp.slice(s![..$x.len()]).to_owned()
            }
        }
    };
}

#[macro_export]
macro_rules! fftfreq {
    ($x:expr, $t:expr) => {
        match ft::zero_pad(&$x) {
            None => {
                ft::fftfreq($x.len(), $t)
            }
            Some(x) => {
                ft::fftfreq(x.len(), $t) // Not sure if this is correct
            }
        }
    };
}

#[cfg(test)]
mod tests {

    #[test]
    fn dft_sine() {
        std::unimplemented!();
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


