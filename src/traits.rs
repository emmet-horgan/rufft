use num_traits::{NumAssign, Float, FloatConst, AsPrimitive};
use num_complex::Complex;
use std::iter::{FromIterator, IntoIterator};
use std::ops::IndexMut;
use crate::fft;

pub trait Collection<T>
where 
    Self: IntoIterator<Item = T> + FromIterator<T> + Clone,
    <Self as IntoIterator>::IntoIter: ExactSizeIterator,
    for<'a> &'a Self: CollectionRef<'a, T>,
    // Compiles if the below is included
    //for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
{}

pub trait CollectionRef<'a, T>
where 
    T: 'a,
    Self: IntoIterator<Item = &'a T> + Clone,
    <Self as IntoIterator>::IntoIter: ExactSizeIterator,
{}

impl<C, F> Collection<F> for C 
where
    F: Float + FloatConst + NumAssign + 'static,
    C: IntoIterator<Item = F> + FromIterator<F> + Clone,
    <C as IntoIterator>::IntoIter: ExactSizeIterator,
    for<'a> &'a C: CollectionRef<'a, F>,
    //for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
{}

impl<'a, C, F> CollectionRef<'a, F> for &'a C 
where
    F: Float + FloatConst + NumAssign + 'static,
    &'a C: IntoIterator<Item = &'a F> + Clone,
    <&'a C as IntoIterator>::IntoIter: ExactSizeIterator
{}

pub trait Fft<F: Float + FloatConst + NumAssign + 'static>
where 
    Self: FromIterator<F> + Clone,
    for<'a> &'a Self: IntoIterator<Item = &'a F>,
    for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
    usize: AsPrimitive<F>,
{   
    fn fft_ct<C>(&self) -> C
    where
        C: FromIterator<Complex<F>> + IndexMut<usize, Output = Complex<F>>
    {
        fft::fft_ct::<Self, C, F>(self)
    }
}

impl<C, F> Fft<F> for C
where 
    Self: IntoIterator<Item = F> + FromIterator<F> + Clone,
    <Self as IntoIterator>::IntoIter: ExactSizeIterator,
    for<'a> &'a Self: IntoIterator<Item = &'a F>,
    for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>
{}


//mpl<'a, F> CollectionRef<'a, F> for &'a Vec<F>
//here
//   F: Float + FloatConst + NumAssign + 'static,
//}

//pub trait FloatCollection<F: Float + FloatConst + NumAssign + 'static>
//where
//    Self: IntoIterator<Item = F, IntoIter = <Self as FloatCollection<F>>::IntoIter>,
//    //<Self as IntoIterator>::IntoIter: ExactSizeIterator,
//    Self: FromIterator<F> + Clone,
//    for<'a> &'a Self: IntoIterator<Item = &'a F, IntoIter = <Self as FloatCollection<F>>::IntoIter>,
//    for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
//{
//    type IntoIter: ExactSizeIterator;
//    fn len(&self) -> usize {
//        self.into_iter().len()
//    }
//
//    fn float_collection_type(&self) {
//        println!("I implement the trait");
//    }
//}
//
//impl<C, F> FloatCollection<F> for C
//where 
//    F: Float + FloatConst + NumAssign + 'static,
//    C: IntoIterator<Item = F> + FromIterator<F> + Clone,
//    <C as IntoIterator>::IntoIter: ExactSizeIterator,
//    for<'a> &'a C: IntoIterator<Item = &'a F>,
//    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
//{
//    type IntoIter = <C as IntoIterator>::IntoIter;
//}

pub trait TyEq
where
    Self: From<Self::Type> + Into<Self::Type>,
    Self::Type: From<Self> + Into<Self>,
{
    type Type;
}

impl<T> TyEq for T {
    type Type = T;
}

pub trait HasInnerFloat {
    type InnerFloat: Float + FloatConst + NumAssign + 'static;
}   

impl<T:'static + Float + FloatConst + NumAssign> HasInnerFloat for Complex<T> {
    type InnerFloat = T;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_collection_trait() {
        let float_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let float_vec_ref = &float_vec;
        //float_vec.print();
        //float_vec_ref.print();

        //float_vec.float_collection_type();
        //let boxed = Box::new(float_vec);
    }
}