use num_traits::{ Float, FloatConst, NumAssign, AsPrimitive, FromPrimitive };
use crate::traits::Iterable;

pub struct SincBuilder<F> 
where 
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    fs: F,
    f: F,
    offset: F,
    norm: bool
}

impl<F> SincBuilder<F> 
where 
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fs(mut self, fs: F) -> Self {
        self.fs = fs;
        self
    }

    pub fn freq(mut self, f: F) -> Self {
        self.f = f;
        self
    }

    pub fn offset(mut self, offset: F) -> Self {
        self.offset = offset;
        self
    }

    pub fn norm(mut self) -> Self {
        self.norm = true;
        self
    }

    pub fn build_iter(self) -> SincIterator<F> {
        SincIterator {
            fs: self.fs,
            f: self.f,
            offset: self.offset,
            norm: self.norm,
            i: 0
        }
    }

    pub fn build_collection(self) -> SincIterator<F> {
        SincIterator {
            fs: self.fs,
            f: self.f,
            offset: self.offset,
            norm: self.norm,
            i: 0
        }
    }
}

impl<F> Default for SincBuilder<F> 
where 
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>
        + FromPrimitive,
    usize: AsPrimitive<F>
{
    fn default() -> Self {
        Self {
            fs: F::from_f64(10.0).unwrap(),
            f: F::one(),
            offset: F::zero(),
            norm: false
        }
    }
}

pub struct SincIterator<F> 
where 
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    fs: F,
    f: F,
    offset: F,
    i: usize,
    norm: bool
}


impl<F> SincIterator<F>
where
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{

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

    pub fn bound<C>(self, length: usize) -> Sinc<F, C>
    where 
        for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>
    {
        self.take(length).collect()
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
        let phase = (F::TAU() * self.f * t) + self.offset;
        if phase == F::zero() {
            Some(F::one())
        } else {
            if self.norm {
                Some(Self::sinc_norm(phase))
            } else {
                Some(Self::sinc(phase))
            }
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
        let mut builder = SincBuilder::<F>::new();

        let iter = SincBuilder::new().freq(f).fs(fs);
        Self (
            iter.take((duration * fs).floor().as_()).collect()
        )
    }
}

super::impl_traits!(Sinc);

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::*;
    use ndarray::prelude::*;

    #[test]
    fn test_collectable() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sine: Sinc<f64, Vec<_>> = samples.iter().cloned().collect();
        for (s, sine) in samples.iter().zip(sine.iter()) {
            println!("{} == {}", s, sine);
            assert_eq!(s, sine);
        }
    }

    #[test]
    fn test_iterable() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sine: Sinc<f64, Vec<_>> = Sinc::from_samples(samples.clone());

        for (s, sine) in samples.iter().zip(sine.iter()) {
            println!("{} == {}", s, sine);
            assert_eq!(s, sine);
        }
    }

    #[test]
    fn test_plotting() {
        let duration = 100.0;
        let fs = 1024.0;
        let n = (duration * fs) as usize;
        let f = 5.0;

        let time: Vec<_> = (0..n).map(|i| i as f64 / fs).collect();
        let waveform = Sinc::<f64, Array1<_>>::from_duration(fs, f, duration);
        let fft: Vec<_> = crate::fft::czt::fft::<_, _, Vec<_>>(&waveform).iter().map(|&x| x.norm()).collect();
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