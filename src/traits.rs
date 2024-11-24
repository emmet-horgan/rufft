use num_traits::{ NumAssign, Float, FloatConst, AsPrimitive };
use num_complex::Complex;
use core::{ ops::IndexMut, ops::Deref };
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

#[cfg(feature = "std")]
impl<T> Iterable for Vec<T>
where 
    for<'c> T: 'c,
    T: Clone,
{
    type Item<'c> = &'c T
    where
        T: 'c;
    type OwnedItem = T;

    type Iterator<'c> = core::slice::Iter<'c, T>
    where
        T: 'c;
    
    fn iter<'c>(&'c self) -> Self::Iterator<'c> {
        self.as_slice().iter()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(all(feature = "ndarray", feature = "std"))]
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
    usize: AsPrimitive<F>,
{   
    fn fft<C>(&self) -> C
    where 
        for<'c> C: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c Complex<F>>,
        C: IndexMut<usize, Output = Complex<F>>,
        usize: AsPrimitive<F>
    {
        let n = self.len();
        if n.is_power_of_two() {
            fft::ct::fft::<F, Self, C>(self)
        } else {
            fft::czt::fft::<F, Self, C>(self)
        }
    }
}

impl<C, F> Fft<F> for C
where 
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>
{}

