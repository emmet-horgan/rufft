use num_traits::{NumAssign, Float, FloatConst};
use num_complex::Complex;
use std::iter::{FromIterator, IntoIterator};

pub trait FloatCollection<F: Float + FloatConst + NumAssign + 'static>
where
    Self: IntoIterator<Item =F>,
    <Self as IntoIterator>::IntoIter: ExactSizeIterator,
    Self: FromIterator<F> + Clone,
    for<'a> &'a Self: IntoIterator<Item = &'a F>,
    for<'a> <&'a Self as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.into_iter().len()
    }
}

impl<C, F> FloatCollection<F> for C
where 
    F: Float + FloatConst + NumAssign + 'static,
    C: IntoIterator<Item = F> + FromIterator<F> + Clone,
    <C as IntoIterator>::IntoIter: ExactSizeIterator,
    for<'a> &'a C: IntoIterator<Item = &'a F>,
    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
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
