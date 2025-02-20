use crate::traits::FloatIterable;
use super::SamplingError;
use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive };
use core::fmt::Debug;
use core::ops::{ Deref, DerefMut };

#[derive(Default)]
pub struct SineBuilder<F>
where 
    F: Float + FloatConst + NumAssign + AsPrimitive<usize> + 'static 
        + Default + Debug,
    usize: AsPrimitive<F>
{
    fs: Option<F>,
    f: Option<F>,
    phase_offset: Option<F>,
    bias: Option<F>,
    amplitude: Option<F>
}

impl<F> SineBuilder<F>
where
    F: Float + FloatConst + NumAssign + AsPrimitive<usize> + 'static 
        + Default + Debug,
    usize: AsPrimitive<F>
{
    pub fn new() -> Self {
        Default::default()
    }

    pub fn fs(mut self, fs: F) -> Self {
        self.fs = Some(fs);
        self
    }

    pub fn f(mut self, f: F) -> Self {
        self.f = Some(f);
        self
    }

    pub fn phase_offset(mut self, offset: F) -> Self {
        self.phase_offset = Some(offset);
        self
    }

    pub fn bias(mut self, bias: F) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn amplitude(mut self, amplitude: F) -> Self {
        self.bias = Some(amplitude);
        self
    }

    pub fn build(self) -> Result<SineIterator<F>, SamplingError<F>> {
        let two = F::one() + F::one();
        let fs = self.fs.ok_or(
            SamplingError::<F>::MissingSamplingFreq
        )?;
        let f = self.f.ok_or(
            SamplingError::<F>::MissingFreq
        )?;
        if fs < (two * f) {
            return Err(SamplingError::ShannonError { f: f, fs: fs })
        }
        let amplitude = match self.amplitude {
            Some(x) => x,
            None => F::one()
        };
        let bias = match self.bias {
            Some(x) => x,
            None => F::zero()
        };
        let phase_offset = match self.phase_offset {
            Some(x) => x,
            None => F::zero()
        };
        Ok(SineIterator {
            f: f,
            fs: fs,
            phase_offset: phase_offset,
            bias: bias,
            amplitude: amplitude,
            i: 0
        })
    }
}

pub struct SineIterator<F>
where 
    F: Float + FloatConst + NumAssign + AsPrimitive<usize> + 'static,
    usize: AsPrimitive<F>
{
    fs: F,
    f: F,
    phase_offset: F,
    bias: F,
    amplitude: F,
    i: usize
}

impl<F> Iterator for SineIterator<F>
where
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        let t = self.i.as_() / self.fs;
        self.i += 1;
        Some(
            (((F::TAU() * self.f * t) + self.phase_offset).sin() * self.amplitude) + self.bias
        )
    }
}

pub struct Sine<F, C>
where 
    F: Float + FloatConst + NumAssign + 'static,
    C: FloatIterable<F> //+ FromIterator<F>
{
    inner: C,
    marker: core::marker::PhantomData<F>
}

impl<F, C> Deref for Sine<F, C> 
where 
    F: Float + FloatConst + NumAssign + 'static,
    C: FloatIterable<F> //+ FromIterator<F>
{
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F, C> DerefMut for Sine<F, C> 
where 
    F: Float + FloatConst + NumAssign + 'static,
    C: FloatIterable<F> //+ FromIterator<F>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// Implement FromIterator for Sine<F, C>
impl<F, C> FromIterator<F> for Sine<F, C>
where
    F: Float + FloatConst + NumAssign + 'static,
    C: FromIterator<F> + FloatIterable<F>,
{
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Sine {
            inner: C::from_iter(iter),
            marker: core::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collectable() {
        //let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let iter = SineBuilder::new()
            .f(100.0)
            .fs(400.0)
            .build()
            .unwrap();
        
        let _sine: Sine<f64, Vec<_>> = iter.take(200).collect();
        
    }
}