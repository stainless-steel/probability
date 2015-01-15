//! A probability-theory toolbox.

#![allow(unstable)]

#[cfg(test)]
#[macro_use]
extern crate assert;

#[cfg(test)]
extern crate test;

extern crate special;

use std::rand::Rng;

pub mod distributions;

/// An interface for a probability distribution.
pub trait Distribution {
    type Item;

    /// Compute the cumulative distribution function (CDF) at point `x`.
    fn cdf(&self, x: Self::Item) -> f64;

    /// Compute the inverse of the cumulative distribution function at
    /// probability `p`.
    fn inv_cdf(&self, p: f64) -> Self::Item;

    /// Draw a random sample.
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Item;
}

/// A means of drawing a sequence of samples from a probability distribution.
///
/// # Example
///
/// ```
/// #![allow(unstable)]
///
/// use std::rand::thread_rng;
/// use probability::Sampler;
/// use probability::distributions::Uniform;
///
/// let uniform = Uniform::new(0.0, 1.0);
/// let samples = Sampler(&uniform, &mut thread_rng()).take(10).collect::<Vec<_>>();
/// ```
pub struct Sampler<D, R>(pub D, pub R);

impl<'a, T, D, R> Iterator for Sampler<&'a D, &'a mut R>
    where D: Distribution<Item=T>, R: Rng {

    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
