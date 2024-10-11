use num_traits::{ AsPrimitive, Float, FloatConst, NumAssign };
use crate::traits::Iterable;

macro_rules! impl_traits {
    ($($name:ident),*) => {
        $(
            impl <F, C> FromIterator<F> for $name<F, C>
            where 
                for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
                F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
                usize: AsPrimitive<F>
            {
                fn from_iter<I: IntoIterator<Item = F>>(iter: I) -> Self {
                    Self (
                        iter.into_iter().collect()
                    )
                }

            }

            impl<F, C> Iterable for $name<F, C>
            where 
                for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
                F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
                usize: AsPrimitive<F>
            {
                type OwnedItem = F;
                type Item<'c> = &'c F;
                type Iterator<'c> = <C as Iterable>::Iterator<'c>;
            
                fn iter<'c>(&'c self) -> Self::Iterator<'c> {
                    self.0.iter()
                }
            }
        )*
    };
}

pub struct SineIterator<F> 
where 
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    fs: F,
    f: F,
    offset: F,
    i: usize
}

impl<F> SineIterator<F>
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
        Some(((F::TAU() * self.f * t) + self.offset).sin())
    }
}

#[derive(Clone)]
pub struct Sine<F, C>(C)
where 
    for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>;


impl<F, C> Sine<F, C>
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
        Self::from_duration_with_offset(fs, f, duration, F::zero())
    }

    pub fn from_duration_with_offset(fs: F, f: F, duration: F, offset: F) -> Self {
        let iter = SineIterator::from_frequency_with_offset(fs, f, offset);
        Self (
            iter.take((duration * fs).floor().as_()).collect()
        )
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


pub fn triangular() {
    std::unimplemented!();
}

pub fn square() {
    std::unimplemented!();
}

pub fn sawtooth() {
    std::unimplemented!();
}

pub fn pulse() {
    std::unimplemented!();
}

pub fn multi_tone() {
    std::unimplemented!();
}

impl_traits!(Sine, Sinc);

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::*;
    use ndarray::prelude::*;

    #[test]
    fn test_collectable() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sine: Sine<f64, Vec<_>> = samples.iter().cloned().collect();
        for (s, sine) in samples.iter().zip(sine.iter()) {
            println!("{} == {}", s, sine);
            assert_eq!(s, sine);
        }
    }

    #[test]
    fn test_iterable() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sine: Sine<f64, Vec<_>> = Sine::from_samples(samples.clone());

        for (s, sine) in samples.iter().zip(sine.iter()) {
            println!("{} == {}", s, sine);
            assert_eq!(s, sine);
        }
    }

    #[test]
    fn test_plotting() {
        let duration = 2.0;
        let fs = 1024.0;
        let n = (duration * fs) as usize;
        let f = 5.0;

        let time: Vec<_> = (0..n).map(|i| i as f64 / fs).collect();
        let waveform = Sine::<f64, Array1<_>>::from_duration(fs, f, duration);
        let fft: Vec<_> = crate::fft::ct::fft::<_, _, Vec<_>>(&waveform).iter().map(|&x| x.norm()).collect();
        let freqs: Vec<_> = crate::fft::fftfreq_balanced(n, 1.0 / fs);
        // Create a drawing area (800x600 pixels)
        let drawing_area = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
        drawing_area.fill(&WHITE).unwrap();

        // Set up the chart
        let mut chart = ChartBuilder::on(&drawing_area)
            .caption("Simple Plot", ("sans-serif", 50).into_font())
            .margin(20)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..10.0, -1.0..1.0).unwrap();

        chart.configure_mesh().draw().unwrap();

        // Plot the data
        chart
            .draw_series(LineSeries::new(
                freqs.iter().zip(fft.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            )).unwrap()
            .label("sin(2t)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels().draw().unwrap();

        }
}

