use super::{ 
    Iterable, 
    FloatIterable, FloatIterableMut,
    ComplexFloatIterable, ComplexFloatIterableMut,
    ExtendableIterable
};
use crate::itertools::{
    pad_clone, pad_inplace, complex,
    zero_pad_clone, zero_pad_inplace, 
    pad_to_nearest_power_of_two_clone, pad_to_nearest_power_of_two_inplace
};
use crate::fft;
use num_traits::{ NumAssign, Float, FloatConst, AsPrimitive };

pub trait RealSignal<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: FloatIterable<F>
{

}

pub trait RealSignalMut<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: FloatIterableMut<F>
{

}

pub trait RealSignalExtendable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: RealSignalMut<F> + ExtendableIterable
{

}

pub trait ComplexSignal<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: ComplexFloatIterable<F>
{

}

pub trait ComplexSignalMut<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: ComplexFloatIterableMut<F>
{

}

pub trait ComplexSignalExtendable<F>
where 
    F: Float + FloatConst + NumAssign + 'static,
    Self: ComplexSignalMut<F> + ExtendableIterable
{

}