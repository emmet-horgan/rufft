pub(crate) mod float;
pub use float::*;
use num_traits::{ NumAssign, Float, FloatConst, AsPrimitive };
use num_complex::Complex;
use core::ops::{ Deref, DerefMut, IndexMut };
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
    /// `Vec<f64>`
    type Item<'collection>
    where
        Self: 'collection;

    type MutItem<'collection>: DerefMut<Target = Self::OwnedItem>
    where 
        Self: 'collection;

    /// The iterator produced by the collection
    type Iterator<'collection>: ExactSizeIterator<Item = Self::Item<'collection>>
        + DoubleEndedIterator<Item = Self::Item<'collection>>
    where
        Self: 'collection;

    type MutIterator<'collection>: ExactSizeIterator<Item = Self::MutItem<'collection>>
        + DoubleEndedIterator<Item = Self::MutItem<'collection>>
    where
        Self: 'collection;

    /// Create an iterator from the collection over `Self::Item` types
    fn iter<'c>(&'c self) -> Self::Iterator<'c>;

    /// Create a mutable iterator from the collection over `Self::MutItem` types
    fn iter_mut<'c>(&'c mut self) -> Self::MutIterator<'c>;

    /// Return the length of the collection. Default implementation provided based
    /// on the fact that an `ExactSizeIterator` implementation is required but 
    /// may be more performant to override the implementation
    #[inline]
    fn len(&self) -> usize {
        self.iter().len()
    }
}

/// Extendable iterable trait that can be implemented on dynamic collect types to 
/// push values onto the collection
pub trait ExtendableIterable
where 
    Self: Iterable,
    for<'c> Self::Item<'c>: Deref<Target = Self::OwnedItem>,
    for<'c> Self::MutItem<'c>: DerefMut<Target = Self::OwnedItem>,
    Self::OwnedItem: Clone
{
    /// Push a `<Self as Iterable>::OwnedItem` type to the collection
    fn push(&mut self, item: Self::OwnedItem);

    /// Extend the collection from a slice of `<Self as Iterable>::OwnedItem` 
    /// types. Default implementation iterates and uses the `push` method
    fn extend_from_slice(&mut self, other: &[Self::OwnedItem]) {
        for x in other {
            self.push(x.clone())
        }
    }
}

#[cfg(feature = "std")]
impl<T> ExtendableIterable for Vec<T>
where 
    T: Clone,
    for<'c> T: 'c
{
    fn push(&mut self, item: Self::OwnedItem) {
        self.push(item);
    }

    fn extend_from_slice(&mut self, other: &[Self::OwnedItem]) {
        self.extend_from_slice(other);
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
    type MutItem<'c> = &'c mut T;
    type OwnedItem = T;

    type Iterator<'c> = core::slice::Iter<'c, T>
    where
        T: 'c;
    
    type MutIterator<'c> = core::slice::IterMut<'c, T>
    where
        T: 'c;
    
    fn iter<'c>(&'c self) -> Self::Iterator<'c> {
        self.as_slice().iter()
    }

    fn iter_mut<'c>(&'c mut self) -> Self::MutIterator<'c> {
        self.as_mut_slice().iter_mut()
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
    type MutItem<'c> = &'c mut T;
    type OwnedItem = T;
    type Iterator<'c> = ndarray::iter::Iter<'c, T, ndarray::Ix1>
    where
        T: 'c;
    type MutIterator<'c> = ndarray::iter::IterMut<'c, T, ndarray::Ix1>;

    fn iter<'c>(&'c self) -> Self::Iterator<'c> {
        self.iter()
    }

    fn iter_mut<'c>(&'c mut self) -> Self::MutIterator<'c> {
        self.iter_mut()
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
    for<'c> Self: Iterable<OwnedItem = F, Item<'c> = &'c F, MutItem<'c> = &'c mut F>,
    usize: AsPrimitive<F>,
{   
    fn fft<C>(&self) -> C
    where 
        for<'c> C: Iterable<
            OwnedItem = Complex<F>, 
            Item<'c> = &'c Complex<F>,
            MutItem<'c> = &'c mut Complex<F>
        >,
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
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F, MutItem<'c> = &'c mut F>,
    F: Float + FloatConst + NumAssign + 'static,
    usize: AsPrimitive<F>
{}
