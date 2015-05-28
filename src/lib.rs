//! A probability-theory toolbox.

#[cfg(test)]
extern crate assert;

extern crate num;
extern crate rand;
extern crate special;

pub use ::rand::Rng as Generator;
pub use ::rand::thread_rng as generator;

pub mod distributions;

/// An interface for a probability distribution.
pub trait Distribution {
    type Item;

    /// Get the expected value `E[X]` of the distribution.
    fn mean(&self) -> f64;

    /// Get the variance `var[X]` of the distribution.
    fn var(&self) -> f64;

    /// Get the standard deviation `sqrt(var[X])` of the distribution.
    fn sd(&self) -> f64;

    /// Get the median of the distribution.
    fn median(&self) -> f64;

    /// Return a vector of all the modes for the distribution.
    fn modes(&self) -> Vec<f64>;

    /// Return the skewness of the distribution.
    fn skewness(&self) -> f64;

    /// Return the excess kurtosis of the distribution.
    fn kurtosis(&self) -> f64;

    /// Return the differential entropy of the distribution (measured in nats).
    fn entropy(&self) -> f64;

    /// Compute the cumulative distribution function (CDF) at point `x`.
    fn cdf(&self, x: Self::Item) -> f64;

    /// Compute the inverse of the cumulative distribution function at
    /// probability `p`.
    fn inv_cdf(&self, p: f64) -> Self::Item;

    /// Compute the probability density function (PDF) at point `x`.
    fn pdf(&self, x: Self::Item) -> f64;

    /// Draw a random sample.
    fn sample<G: Generator>(&self, generator: &mut G) -> Self::Item;
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
    where D: Distribution<Item=T>, G: Generator {

    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
