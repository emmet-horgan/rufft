use ndarray;
use ndarray::prelude::*;
use std::ops::Index;
use std::iter::Iterator;
/*
use crate::traits::{Signal, SignalIterator};

pub struct TimeSignal<T, U> {
    data: Array1<T>,
    fs: U
}

impl<T, U> TimeSignal<T, U> {
    pub fn new(data: Array1<T>, fs: U) -> Self {
        TimeSignal {data, fs}
    }

    //pub fn iter(&self) -> TimeSignalIterator<'_, T> {
    //    TimeSignalIterator { data_iter: self.data.iter() }
    //}
}
impl<T, U> Signal for TimeSignal<T, U> {
    type ArrayType = Array1<T>;
    type ElementType = T;
    type IndexType = U;

    fn iter(&self) ->TimeSignalIterator<'_, T>{
        TimeSignalIterator { data_iter: self.data.iter() }
    }

    fn dim(&self) -> U {
        U::from(value)
    }
}
impl<T, U> Index<usize> for TimeSignal<T, U> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
    
}

pub struct TimeSignalIterator<'a, T> {
    data_iter: ndarray::iter::Iter::<'a, T, Ix1>
}

impl<'a, T> Iterator for TimeSignalIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.data_iter.next()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use crate::signal;
    #[test]
    fn time_signal() {
        let x: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let fs: f64 = 32.0;

        let sig = signal::TimeSignal::new(x, fs);
        for i in sig.iter() {
            println!("i = {i}");
        }
        assert!(true);
    }
}
*/