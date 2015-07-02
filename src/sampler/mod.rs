//! Samplers of random numbers.

use distribution::Distribution;
use random::Source;

/// A means of drawing a sequence of independent samples.
pub struct Independent<D, S>(pub D, pub S);

impl<'a, T, D, S> Iterator for Independent<&'a D, &'a mut S>
    where D: Distribution<Value=T>, S: Source
{
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        Some(self.0.sample(self.1))
    }
}
