//! Probability distributions.

use std::rand::Rng;

pub use self::beta::Beta;
pub use self::gaussian::Gaussian;
pub use self::uniform::Uniform;

mod beta;
mod gaussian;
mod uniform;

/// An interface for a probability distribution.
pub trait Distribution<T> {
    /// Computes the cumulative distribution function (CDF) at point `x`.
    fn cdf(&self, x: T) -> f64;

    /// Computes the inverse of the cumulative distribution function at
    /// probability `p`.
    fn inv_cdf(&self, p: f64) -> T;

    /// Draws a random sample.
    fn sample<R: Rng>(&self, rng: &mut R) -> T;
}

/// A means of drawing a sequence of samples from a probability distribution.
///
/// # Example
///
/// ```
/// use prob::dist::{Sampler, Uniform};
///
/// let mut rng = std::rand::task_rng();
/// let uniform = Uniform::new(0.0, 1.0);
/// let samples = Sampler(&uniform, &mut rng).take(10).collect::<Vec<_>>();
/// ```
pub struct Sampler<D, R>(pub D, pub R);

impl<'a, T, D, R> Iterator<T> for Sampler<&'a D, &'a mut R>
    where D: Distribution<T>, R: Rng {

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
