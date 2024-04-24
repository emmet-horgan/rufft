use std::convert::Into;
use std::ops::{Add, Div, Mul, Sub};
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

pub trait FloatVal: Num + NumAssignOps + Float + FloatConst + 'static {}

pub trait SigVal<T: FloatVal>: Num + NumAssignOps + 'static 
where 
    Self: AsPrimitive<T>
{}

impl FloatVal for f64 {}
impl FloatVal for f32 {}

impl SigVal<f64> for f64 {}
impl SigVal<f32> for f32 {}

impl SigVal<f64> for i64 {}
impl SigVal<f32> for i32 {}
