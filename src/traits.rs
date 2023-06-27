pub trait Stats {
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn stdev(&self) -> f64;
    fn skewness(&self) -> f64;
    fn kurtosis(&self) -> f64;
    fn histogram(&self);
}
