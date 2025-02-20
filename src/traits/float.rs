use super::{ Iterable, ExtendableIterable };
use num_traits::{ Float, FloatConst, NumAssign };
use num_complex::Complex;
use core::ops::Deref;
use crate::itertools::{
    complex,
    pad_clone, pad_to_nearest_power_of_two_clone, zero_pad_clone, 
    zero_pad_to_nearest_power_of_two_clone,
    pad_inplace, pad_to_nearest_power_of_two_inplace, zero_pad_inplace, 
    zero_pad_to_nearest_power_of_two_inplace,
};

pub trait FloatIterable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> Self: Iterable<OwnedItem = F, Item<'c> = &'c F, MutItem<'c> = &'c mut F>,
{
    fn pad_clone(&self, pad: F, n: usize) -> Self {
        pad_clone(self, pad, n)
    }

    fn pad_to_nearest_power_of_two_clone(&self, pad: F) -> Self {
        pad_to_nearest_power_of_two_clone(self, pad)
    }

    fn zero_pad_clone(&self, n: usize) -> Self {
        zero_pad_clone(self, n)
    }

    fn zero_pad_to_nearest_power_of_two_clone(&self) -> Self {
        zero_pad_to_nearest_power_of_two_clone(self)
    }
}

impl<F, T> FloatIterable<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<OwnedItem = F, Item<'c> = &'c F, MutItem<'c> = &'c mut F>
{}

pub trait FloatIterableExtendable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: ExtendableIterable + FloatIterable<F>
{
    fn pad_inplace(&mut self, pad: F, n: usize) {
        pad_inplace(self, pad, n);
    }

    fn pad_to_nearest_power_of_two_inplace(&mut self, pad: F) {
        pad_to_nearest_power_of_two_inplace(self, pad);
    }

    fn zero_pad_inplace(&mut self, n: usize) {
        zero_pad_inplace(self, n);
    }

    fn zero_pad_to_nearest_power_of_two_inplace(&mut self) {
        zero_pad_to_nearest_power_of_two_inplace(self)
    }
}

impl<F, T> FloatIterableExtendable<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<OwnedItem = F, Item<'c> = &'c F, MutItem<'c> = &'c mut F>,
    T: ExtendableIterable
{}

pub trait ComplexFloatIterable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> Self: Iterable<
        OwnedItem = Complex<F>, 
        Item<'c> = &'c Complex<F>,
        MutItem<'c> = &'c mut Complex<F>
    >
{
    fn pad_clone(&self, pad: Complex<F>, n: usize) -> Self {
        complex::pad_clone(self, pad, n)
    }

    fn pad_to_nearest_power_of_two_clone(&self, pad: Complex<F>) -> Self {
        complex::pad_to_nearest_power_of_two_clone(self, pad)
    }

    fn zero_pad_clone(&self, n: usize) -> Self {
        complex::zero_pad_clone(self, n)
    }

    fn zero_pad_to_nearest_power_of_two_clone(&self) -> Self {
        complex::zero_pad_to_nearest_power_of_two_clone(self)
    }
}

impl<F, T> ComplexFloatIterable<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<
        OwnedItem = Complex<F>, 
        Item<'c> = &'c Complex<F>,
        MutItem<'c> = &'c mut Complex<F>
    >
{}

pub trait ComplexFloatIterableExtendable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: ExtendableIterable + ComplexFloatIterable<F>
{
    fn pad_inplace(&mut self, pad: Complex<F>, n: usize) {
        complex::pad_inplace(self, pad, n);
    }

    fn pad_to_nearest_power_of_two_inplace(&mut self, pad: Complex<F>) {
        complex::pad_to_nearest_power_of_two_inplace(self, pad);
    }

    fn zero_pad_inplace(&mut self, n: usize) {
        complex::zero_pad_inplace(self, n);
    }

    fn zero_pad_to_nearest_power_of_two_inplace(&mut self) {
        complex::zero_pad_to_nearest_power_of_two_inplace(self)
    }
}

impl<F, T> ComplexFloatIterableExtendable<F> for T 
where
    F: Float + FloatConst + NumAssign + 'static,
    for<'c> T: Iterable<
        OwnedItem = Complex<F>, 
        Item<'c> = &'c Complex<F>, 
        MutItem<'c> = &'c mut Complex<F>
    >,
    T: ExtendableIterable
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