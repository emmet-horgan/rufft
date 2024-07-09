use num_traits::{AsPrimitive, Num, NumAssignOps};
use num_traits::float::{Float, FloatConst};

pub trait FloatVal: NumAssignOps + Float + FloatConst + 'static {}

pub trait SigVal<T: FloatVal>: Num + NumAssignOps + 'static 
where 
    Self: AsPrimitive<T>
{}

//pub trait Signal<T: FloatVal, I: ExactSizeIterator>: ExactSizeIterator + SigVal<T>
//where 
//    I::Item: SigVal<T>
//{}

//pub trait Signal<T: FloatVal, I: Iterator>: SigVal<T>
//where
//    I::Item: SigVal<T>
//{}

//pub struct SignalIter<T: FloatVal, I> {
//    marker: std::marker::PhantomData<T>,
//    iter: I
//}

//impl<T, I> Iterator for SignalIter<T, I> 
//where
//    T: FloatVal,
//    I: Iterator,
//    I::Item: SigVal<T>,
//{
//    type Item = I::Item;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        self.iter.next()
//    }
//}
//
//impl<T, I> Signal for SignalIter<T, I>
//where
//    T: FloatVal,
//    I: Iterator,
//    I::Item: SigVal<T>,
//{}
//pub trait SignalIter<T: FloatVal, I>
//where 
//    I: Iterator,
//    I::Item: SigVal<T>,
//{}

impl FloatVal for f64 {}
impl FloatVal for f32 {}

impl SigVal<f64> for f64 {}
impl SigVal<f32> for f32 {}

impl SigVal<f64> for i64 {}
impl SigVal<f32> for i32 {}
