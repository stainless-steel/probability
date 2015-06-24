//! Samplers of random numbers.

use distribution::Distribution;
use generator::Generator;

/// A means of drawing a sequence of independent samples.
///
/// ## Example
///
/// ```
/// use probability::prelude::*;
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
