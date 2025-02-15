use num_traits::{ NumAssign, Float, FloatConst, AsPrimitive };
use num_complex::Complex;
use core::ops::{ IndexMut, Deref };
use crate::fft;

/// Iterable trait to encapsulate collection types which have a length, are 
/// reversible, and are iterable
pub trait Iterable: FromIterator<Self::OwnedItem>
where 
    for<'c> Self::Item<'c>: Deref<Target = Self::OwnedItem>,
    for<'c> Self: 'c,
    Self: Clone,
{
    /// The owned collection item. For example `f64` for `Vec<f64>`
    type OwnedItem;
    
    /// The item produced by the collection's iterator. For example `&f64` for 
    /// `Vec<f64`
    type Item<'collection>
    where
        Self: 'collection;

    /// The iterator produced by the collection
    type Iterator<'collection>: ExactSizeIterator<Item = Self::Item<'collection>>
        + DoubleEndedIterator<Item = Self::Item<'collection>>
    where
        Self: 'collection;

    /// Create an iterator from the collection over `Self::Item` types
    fn iter<'c>(&'c self) -> Self::Iterator<'c>;

    /// Return the length of the collection. Default implementation provided based
    /// on the fact that an `ExactSizeIterator` implementation is required but 
    /// may be more performant to override the implementation
    #[inline]
    fn len(&self) -> usize {
        self.iter().len()
    }
}

pub trait FloatIterable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> Self: Iterable<OwnedItem = F, Item<'c> = &'c F>,
{}

impl<F, T> FloatIterable<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<OwnedItem = F, Item<'c> = &'c F>
{}

pub trait ComplexFloatIterable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> Self: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c Complex<F>>,
{}

impl<F, T> ComplexFloatIterable<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c Complex<F>>
{}

pub trait FloatIterable2d<F> 
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: Iterable,
    for<'c> Self::Item<'c>: Deref<Target = Self::OwnedItem>,
    for<'c> Self::OwnedItem: Iterable<OwnedItem = F, Item<'c> = &'c F>
{
    const DIMS: usize = 2;
}

impl<F, T> FloatIterable2d<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    T: Iterable,
    for<'c> T::Item<'c>: Deref<Target = T::OwnedItem>,
    for<'c> T::OwnedItem: Iterable<OwnedItem = F, Item<'c> = &'c F>
{}

pub trait FloatIterable3d<F> 
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: Iterable,
    for<'c> Self::Item<'c>: Deref<Target = Self::OwnedItem>,
    for<'c> Self::OwnedItem: Iterable,
    for<'c> <Self::OwnedItem as Iterable>::OwnedItem: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    for<'c> <Self::OwnedItem as Iterable>::Item<'c>: Deref<Target = <Self::OwnedItem as Iterable>::OwnedItem>
{
    const DIMS: usize = 3;
}

impl<F, T> FloatIterable3d<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    T: Iterable,
    for<'c> T::Item<'c>: Deref<Target = T::OwnedItem>,
    for<'c> T::OwnedItem: Iterable,
    for<'c> <T::OwnedItem as Iterable>::OwnedItem: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    for<'c> <T::OwnedItem as Iterable>::Item<'c>: Deref<Target = <T::OwnedItem as Iterable>::OwnedItem>
{}

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

/// Trait containing `fft` method which computes the `fft` of the real valued
/// collection type and returns a complex value collection. The return collection
/// type does not need to be the same type as the type the trait is implemented
/// on
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
