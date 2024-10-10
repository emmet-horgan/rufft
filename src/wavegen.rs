use num_traits::{ AsPrimitive, Float, FloatConst, NumAssign };

//macro_rules! impl_traits {
//    ($name:ty, $samples:ident) => {
//        impl<F, C> FromIterator<$F> for $name<F, C>
//        where 
//            for<'c> &'c C: IntoIterator<Item = &'c F>,
//            for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
//            F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
//            usize: AsPrimitive<F>
//        {
//            fn from_iter<I: IntoIterator<Item = F>>(iter: I) -> Self {
//                Self {
//                    $samples: iter.into_iter().collect(),
//                }
//            }
//        }
//
//        impl
//    };
//}

pub struct Sine<F, C>
where 
    for<'c> &'c C: IntoIterator<Item = &'c F>,
    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    pub samples: C,
}

impl<F, C> Sine<F, C>
where 
    C: FromIterator<F> + Clone,
    for<'c> &'c C: IntoIterator<Item = &'c F>,
    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Float + FloatConst + NumAssign + 'static,
    F: AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    pub fn from_samples(samples: C) -> Self {
        Self {
            samples,
        }
    }

    pub fn from_duration(fs: F, f: F, duration: F) -> Self {
        Self::from_duration_with_offset(fs, f, duration, F::zero())
    }

    pub fn from_duration_with_offset(fs: F, f: F, duration: F, offset: F) -> Self {
        let n: usize = (duration * fs).floor().as_();
        Self {
            samples: (0..n).map(|i| {
                let t = i.as_() / fs;
                ((F::TAU() * f * t) + offset).sin()
            }).collect()
        }
    }
}

pub struct Sinc<F, C>
where 
    for<'c> &'c C: IntoIterator<Item = &'c F>,
    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Float + FloatConst + NumAssign + 'static
{
    pub samples: C,
}

impl<F, C> Sinc<F, C>
where 
    C: FromIterator<F> + Clone,
    for<'c> &'c C: IntoIterator<Item = &'c F>,
    for<'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
    F: Float + FloatConst + NumAssign + 'static,
    F: AsPrimitive<usize>,
    usize: AsPrimitive<F>
{
    pub fn from_samples(samples: C) -> Self {
        Self {
            samples,
        }
    }

    pub fn from_duration(fs: F, f: F, duration: F) -> Self {
        let n: usize = (duration * fs).floor().as_();
        Self {
            samples: (0..n).map(|i| {
                let t = i.as_() / fs;
                let phase = F::TAU() * f * t;
                if phase == F::zero() {
                    F::one()
                } else {
                    phase.sin() / phase
                }
            }).collect()
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::*;

    #[test]
    fn test_plotting() {
        let duration = 1000.0;
        let fs = 1000.0;
        let n = (duration * fs) as usize;
        let f = 1.0;

        let time: Vec<_> = (0..n).map(|i| i as f64 / fs).collect();
        let waveform = Sinc::<f64, Vec<_>>::from_duration(fs, f, duration);

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
                time.iter().zip(waveform.samples.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            )).unwrap()
            .label("sin(2t)")
            .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels().draw().unwrap();

        }
}

