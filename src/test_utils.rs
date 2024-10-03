use num_traits::{Float, AsPrimitive};

const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-9;

pub fn nearly_equal<F: Float + 'static>(a: F, b: F) -> bool 
where 
    f64: AsPrimitive<F>
{
    let diff = (a - b).abs();
    if diff <= ABS_TOLERANCE.as_() {
        true
    } else {
        diff / a.max(b).abs() <= REL_TOLERANCE.as_()
    }
}
