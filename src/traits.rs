pub trait Stats {
    type Elements;

    fn mean(&self) -> Self::Elements;
    fn variance(&self) -> Self::Elements;
    fn stdev(&self) -> Self::Elements;
    fn skewness(&self) -> Self::Elements;
    fn kurtosis(&self) -> Self::Elements;
    fn histogram(&self);
}
