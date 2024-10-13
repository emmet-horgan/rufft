use num_traits::{ NumAssign, Float, FloatConst, AsPrimitive };
use num_complex::Complex;
use std::{ ops::IndexMut, ops::Deref };
use crate::fft;

pub trait Iterable: FromIterator<Self::OwnedItem>
where 
    for<'c> Self::Item<'c>: Deref<Target = Self::OwnedItem>,
    for<'c> Self: 'c,
    Self: Clone,
{
    type OwnedItem;
    type Item<'collection>
    where
        Self: 'collection;

    type Iterator<'collection>: ExactSizeIterator<Item = Self::Item<'collection>>
        + DoubleEndedIterator<Item = Self::Item<'collection>>
    where
        Self: 'collection;

    fn iter<'c>(&'c self) -> Self::Iterator<'c>;

    fn len(&self) -> usize {
        self.iter().len()
    }
}

impl<T> Iterable for Vec<T>
where 
    for<'c> T: 'c,
    T: Clone,
{
    type Item<'c> = &'c T
    where
        T: 'c;
    type OwnedItem = T;

    type Iterator<'c> = std::slice::Iter<'c, T>
    where
        T: 'c;
    
    fn iter<'c>(&'c self) -> Self::Iterator<'c> {
        self.as_slice().iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}


#[cfg(feature = "ndarray")]
impl<T> Iterable for ndarray::Array1<T>
where
    for<'c> T: 'c,
    T: Clone,
{
    type Item<'c> = &'c T
    where
        T: 'c;
    type OwnedItem = T;
    type Iterator<'c> = ndarray::iter::Iter<'c, T, ndarray::Ix1>
    where
        T: 'c;

    fn iter<'c>(&'c self) -> Self::Iterator<'c> {
        self.iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}


pub trait Fft<F: Float + FloatConst + NumAssign + 'static>
where 
    for<'c> Self: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    for<'c> Self: 'c,
    usize: AsPrimitive<F>,
{   
    fn fft_ct<C>(&self) -> C
    where
        for<'c> C: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c Complex<F>>,
        for<'c> C: 'c,
        C: IndexMut<usize, Output = Complex<F>>,
    {
        fft::ct::fft::<F, Self, C>(self)
    }

    
}

impl<C, F> Fft<F> for C
where 
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    for<'c> C: 'c,
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
