//! A probability-theory toolbox.

#[cfg(test)]
extern crate assert;

extern crate random;
extern crate special;

use std::cell::RefCell;
use std::rc::Rc;

use random::{Generator, Quantity};
use random::XorshiftPlus;

macro_rules! should(
    ($requirement:expr) => ({
        debug_assert!($requirement, stringify!($requirement))
    });
    ($requirement:expr, $code:expr) => ({
        debug_assert!($code, stringify!($requirement))
    });
);

pub mod distributions;

/// The default generator, which is the Xorshift+ algorithm.
pub struct DefaultGenerator(Rc<RefCell<XorshiftPlus>>);

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

    /// Compute the cumulative distribution function.
    fn cdf(&self, x: Self::Value) -> f64;

    /// Compute the inverse of the cumulative distribution function.
    fn inv_cdf(&self, p: f64) -> Self::Value;

    /// Compute the probability density function.
    fn pdf(&self, x: Self::Value) -> f64;

    /// Draw a sample.
    fn sample<G: Generator>(&self, generator: &mut G) -> Self::Value;
}

impl DefaultGenerator {
    /// Seed the generator.
    #[inline(always)]
    pub fn seed(&mut self, seed: [u64; 2]) -> &mut DefaultGenerator {
        *self.0.borrow_mut() = XorshiftPlus::new(seed);
        self
    }
}

impl Generator for DefaultGenerator {
    #[inline(always)]
    fn read(&mut self) -> u64 {
        self.0.borrow_mut().read()
    }

    #[inline(always)]
    fn next<T: Quantity>(&mut self) -> T {
        self.0.borrow_mut().next()
    }
}

/// Return the default generator.
#[inline(always)]
pub fn generator() -> DefaultGenerator {
    thread_local!(static DEFAULT_GENERATOR: Rc<RefCell<XorshiftPlus>> = {
        Rc::new(RefCell::new(XorshiftPlus::new([42, 69])))
    });
    DefaultGenerator(DEFAULT_GENERATOR.with(|generator| generator.clone()))
}

/// A means of drawing a sequence of samples from a probability distribution.
///
/// ## Example
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
    where D: Distribution<Value=T>, G: Generator
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
