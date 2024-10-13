use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive };
use crate::traits::Iterable;

pub struct SincIterator<F> 
where 
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    fs: F,
    f: F,
    offset: F,
    i: usize
}

impl<F> SincIterator<F>
where
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    pub fn from_frequency_with_offset(fs: F, f: F, offset: F) -> Self{
        Self {
            fs,
            f,
            offset,
            i: 0
        }
    }
    pub fn from_frequency(fs: F, f: F) -> Self {
        Self::from_frequency_with_offset(fs, f, F::zero())
    }

    fn sinc(phase: F) -> F {
        if phase == F::zero() {
            F::one()
        } else {
           phase.sin() / phase
        }
    }

    fn sinc_norm(phase: F) -> F {
        if phase == F::zero() {
            F::one()
        } else {
           (phase * F::PI()).sin() / (phase * F::PI())
        }
    }
}

impl<F> Iterator for SincIterator<F>
where
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        let t = self.i.as_() / self.fs;
        self.i += 1;
        let phase = F::TAU() * self.f * t;
        if phase == F::zero() {
            Some(F::one())
        } else {
            Some(phase.sin() / phase)
        }
    }
}

#[derive(Clone)]
pub struct Sinc<F, C>(C)
where 
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>;


impl<F, C> Sinc<F, C>
where 
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    pub fn from_samples(samples: C) -> Self {
        Self (
            samples,
        )
    }

    pub fn from_duration(fs: F, f: F, duration: F) -> Self {
        let n: usize = (duration * fs).floor().as_();
        Self (
            (0..n).map(|i| {
                let t = i.as_() / fs;
                let phase = F::TAU() * f * t;
                if phase == F::zero() {
                    F::one()
                } else {
                    phase.sin() / phase
                }
            }).collect()
        )
    }
}
