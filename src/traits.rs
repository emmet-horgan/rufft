use std::convert::Into;
use std::usize;
use num_traits::{AsPrimitive, Num, NumAssignOps};
use num_traits::float::{Float, FloatConst};

pub trait Stats {
    type Elements;

    fn mean(&self) -> Self::Elements;
    fn variance(&self) -> Self::Elements;
    fn stdev(&self) -> Self::Elements;
    fn skewness(&self) -> Self::Elements;
    fn kurtosis(&self) -> Self::Elements;
    fn histogram(&self);
} 

mod marker {
    trait AsPrimitiveTo {}
}

pub trait SigVal: 
    Into<f64> + AsPrimitive<f64> + AsPrimitive<f32> + Num + NumAssignOps + Float + FloatConst
{
}

impl SigVal for f64 {
}
impl SigVal for f32 {
}

