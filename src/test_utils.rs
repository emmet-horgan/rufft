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