//! A probability-theory toolbox.

#[cfg(test)] extern crate assert;
extern crate special;

/// A means of drawing a sequence of samples from a probability distribution.
///
/// ## Example
///
/// ```
/// use probability::{Sampler, generator};
/// use probability::distribution::Uniform;
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

macro_rules! should(
    ($requirement:expr) => ({
        debug_assert!($requirement, stringify!($requirement))
    });
    ($requirement:expr, $code:expr) => ({
        debug_assert!($code, stringify!($requirement))
    });
);

pub mod distribution;
pub mod generator;

pub use distribution::Distribution;
pub use generator::Generator;
pub use generator::default as generator;
