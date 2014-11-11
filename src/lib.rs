//! A probability-theory toolbox.

#![feature(phase, tuple_indexing)]

extern crate sfunc;

use std::rand::Rng;

pub mod distributions;

/// An interface for a probability distribution.
pub trait Distribution<T> {
    /// Compute the cumulative distribution function (CDF) at point `x`.
    fn cdf(&self, x: T) -> f64;

    /// Compute the inverse of the cumulative distribution function at
    /// probability `p`.
    fn inv_cdf(&self, p: f64) -> T;

    /// Draw a random sample.
    fn sample<R: Rng>(&self, rng: &mut R) -> T;
}

/// A means of drawing a sequence of samples from a probability distribution.
///
/// # Example
///
/// ```
/// use std::rand::task_rng;
/// use probability::Sampler;
/// use probability::distributions::Uniform;
///
/// let uniform = Uniform::new(0.0, 1.0);
/// let samples = Sampler(&uniform, &mut task_rng()).take(10).collect::<Vec<_>>();
/// ```
pub struct Sampler<D, R>(pub D, pub R);

impl<'a, T, D, R> Iterator<T> for Sampler<&'a D, &'a mut R>
    where D: Distribution<T>, R: Rng {

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
