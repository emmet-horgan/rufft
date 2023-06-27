pub mod fft;
pub mod splinex;
pub mod filterx;
pub mod spectrax;
pub mod wavegen;
pub mod tfa;
pub mod inspect;
pub mod convolvex;
pub mod traits;


pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
