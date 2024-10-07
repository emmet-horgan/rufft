use num_traits::{Float, AsPrimitive};
use num_complex::Complex;

//const ABS_TOLERANCE: f64 = 1e-9;
//const REL_TOLERANCE: f64 = 1e-6;

//pub fn nearly_equal<F: Float + 'static>(a: F, b: F) -> bool 
//where 
//    f64: AsPrimitive<F>
//{
//    let diff = (a - b).abs();
//    if diff <= ABS_TOLERANCE.as_() {
//        true
//    } else {
//        diff / a.max(b).abs() <= REL_TOLERANCE.as_()
//    }
//}

pub fn nearly_equal<F: Float + 'static>(a: F, b: F, rtol: F, atol: F) -> bool 
{
    if a.abs() < atol && b.abs() < atol {
        return (a - b).abs() < atol;
    } else {
        (a - b).abs() <= rtol * b.abs()
    }
    //(a - b).abs() <= atol + rtol * b.abs()
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

    // If magnitudes are near zero, consider them equal
    if mag_a < atol && mag_b < atol {
        return true;
    }
    // Compare the phase only if magnitudes are sufficiently large
    nearly_equal(a.arg(), b.arg(), rtol, atol)
    //let phase_diff = (a.arg() - b.arg()).abs();
    //phase_diff <= atol
}