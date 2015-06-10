//! A probability-theory toolbox.

#[cfg(test)]
extern crate assert;

extern crate rand;
extern crate special;

pub use rand::Rng as Generator;
pub use rand::thread_rng as generator;

pub mod distributions;

/// A probability distribution.
pub trait Distribution {
    type Value;

    /// Compute the expected value.
    fn mean(&self) -> f64;

    /// Compute the variance.
    fn var(&self) -> f64;

    /// Compute the standard deviation.
    #[inline]
    fn sd(&self) -> f64 { self.var().sqrt() }

    /// Compute the skewness.
    fn skewness(&self) -> f64;

    /// Compute the excess kurtosis.
    fn kurtosis(&self) -> f64;

    /// Compute the median.
    fn median(&self) -> f64;

    /// Compute all the modes.
    fn modes(&self) -> Vec<Self::Value>;

    /// Compute the differential entropy (measured in nats).
    fn entropy(&self) -> f64;

    /// Compute the cumulative distribution function (CDF) at point `x`.
    fn cdf(&self, x: Self::Value) -> f64;

    /// Compute the inverse of the cumulative distribution function at
    /// probability `p`.
    fn inv_cdf(&self, p: f64) -> Self::Value;

    /// Compute the probability density function (PDF) at point `x`.
    fn pdf(&self, x: Self::Value) -> f64;

    /// Draw a random sample.
    fn sample<G: Generator>(&self, generator: &mut G) -> Self::Value;
}

/// A means of drawing a sequence of samples from a probability distribution.
///
/// # Example
///
/// ```
/// use probability::generator;
/// use probability::Sampler;
/// use probability::distributions::Uniform;
///
/// let uniform = Uniform::new(0.0, 1.0);
/// let samples = Sampler(&uniform, &mut generator()).take(10).collect::<Vec<_>>();
/// ```
pub struct Sampler<D, G>(pub D, pub G);

impl<'a, T, D, G> Iterator for Sampler<&'a D, &'a mut G>
    where D: Distribution<Value=T>, G: Generator {

    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
