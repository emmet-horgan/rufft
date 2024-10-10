use num_traits::{NumAssign, Float, FloatConst, AsPrimitive};
use num_complex::{ Complex, ComplexFloat };
use std::iter::{FromIterator, IntoIterator};
use std::ops::IndexMut;
use crate::fft;

macro_rules! vec_impl_iterable {
    ($item:ty) => {
        impl Iterable<$item> for Vec<$item> {
            type Iterator<'c> = std::slice::Iter<'c, $item>
            where
                Self: 'c;
        
            fn iter<'c>(&'c self) -> Self::Iterator<'c> {
                self.iter()
            }
        }
    };
}

macro_rules! ndarray_impl_iterable {
    ($item:ty) => {
        impl Iterable<$item> for ndarray::Array1<$item> {
            type Iterator<'c> = ndarray::iter::Iter<'c, $item, ndarray::Ix1>
            where
                Self: 'c;
        
            fn iter<'c>(&'c self) -> Self::Iterator<'c> {
                self.iter()
            }
        }
    };
}


// We define an additional trait to extract the float type from the collection.
pub trait FloatCollection {
    type FloatType: Float + FloatConst + NumAssign + 'static;
    type ComplexType: ComplexFloat<Real = Self::FloatType>;
    type ComplexCollection: Iterable<Self::ComplexType>;
}


//impl<F: Float + FloatConst + NumAssign + 'static> FloatCollection for Vec<F> {
//    type FloatType = F;
//    type ComplexType = Complex<F>;
//    type ComplexCollection = Vec<Complex<F>>;
//}


pub trait Iterable<T> {
    // Type of iterator we return. Will return `Self::Item` elements.
    type Iterator<'c>: ExactSizeIterator<Item = &'c T>
    where
        Self: 'c, T: 'c;

    fn iter<'c>(&'c self) -> Self::Iterator<'c>;
    
    fn len(&self) -> usize {
        self.iter().len()
    }
}

vec_impl_iterable!(f64);
vec_impl_iterable!(Complex<f64>);
vec_impl_iterable!(f32);
vec_impl_iterable!(Complex<f32>);

ndarray_impl_iterable!(f64);
ndarray_impl_iterable!(Complex<f64>);
ndarray_impl_iterable!(f32);
ndarray_impl_iterable!(Complex<f32>);


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
        fft::ct::fft::<Self, C, F>(self)
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
