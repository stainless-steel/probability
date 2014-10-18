//! Probability distributions.

pub use self::uniform::Uniform;

mod uniform;

/// An interface for a probability distribution.
pub trait Distribution<T> {
    /// Computes the cumulative distribution function (CDF) at point `x`.
    fn cdf(&self, x: T) -> f64;

    /// Computes the inverse of the cumulative distribution function at
    /// probability `p`.
    fn inv_cdf(&self, p: f64) -> T;

    /// Draws a random sample.
    fn sample(&self) -> T;
}

/// Provides a means of drawing a sequence of samples from a probability
/// distribution.
///
/// # Example
///
/// ```
/// use prob::dist::{Sampler, Uniform};
/// let samples = Sampler(&Uniform::new(0.0, 1.0)).take(10).collect::<Vec<_>>();
/// ```
pub struct Sampler<D>(pub D);

impl<'a, T, D> Iterator<T> for Sampler<&'a D> where D: Distribution<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample())
    }
}
