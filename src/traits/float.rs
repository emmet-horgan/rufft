use super::Iterable;
use num_traits::{ Float, FloatConst, NumAssign };
use num_complex::Complex;
use core::ops::Deref;

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

pub trait FloatIterableMut<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> Self: Iterable<OwnedItem = F, Item<'c> = &'c mut F>,
{}

impl<F, T> FloatIterableMut<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<OwnedItem = F, Item<'c> = &'c mut F>
{}

pub trait ComplexFloatIterableMut<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> Self: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c mut Complex<F>>,
{}

impl<F, T> ComplexFloatIterableMut<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<OwnedItem = Complex<F>, Item<'c> = &'c mut Complex<F>>
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