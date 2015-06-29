//! Samplers of random numbers.

use distribution::Distribution;
use generator::Generator;

/// A means of drawing a sequence of independent samples.
pub struct Independent<D, G>(pub D, pub G);

impl<'a, T, D, G> Iterator for Independent<&'a D, &'a mut G>
    where D: Distribution<Value=T>, G: Generator
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
