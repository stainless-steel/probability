use distribution::Distribution;
use generator::Generator;

/// A Bernoulli distribution.
#[derive(Clone, Copy)]
pub struct Bernoulli {
    p: f64,
    q: f64,
    pq: f64,
}

impl Bernoulli {
    /// Create a Bernoulli distribution with success probability `p`.
    ///
    /// It should hold that `p > 0` and `p < 1`.
    #[inline]
    pub fn new(p: f64) -> Bernoulli {
        should!(p > 0.0 && p < 1.0);
        Bernoulli { p: p, q: 1.0 - p, pq: p * (1.0 - p) }
    }

    /// Create a Bernoulli distribution with failure probability `q`.
    ///
    /// It should hold that `q > 0` and `q < 1`. This constructor is preferable
    /// when `q` is very small.
    #[inline]
    pub fn new_failprob(q: f64) -> Bernoulli {
        should!(q > 0.0 && q < 1.0);
        Bernoulli { p: 1.0 - q, q: q, pq: (1.0 - q) * q }
    }

    /// Return the success probability.
    #[inline(always)] pub fn p(&self) -> f64 { self.p }

    /// Return the failure probability.
    #[inline(always)] pub fn q(&self) -> f64 { self.q }
}

impl Distribution for Bernoulli {
    type Value = u8;

    #[inline] fn mean(&self) -> f64 { self.p }
    #[inline] fn var(&self) -> f64 { self.pq }

    #[inline]
    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p) / (self.pq).sqrt()
    }

    #[inline]
    fn kurtosis(&self) -> f64 {
        (1.0 - 6.0 * self.pq) / (self.pq)
    }

    #[inline]
    fn median(&self) -> f64 {
        use std::cmp::Ordering::*;
        match self.p.partial_cmp(&self.q) {
            Some(Less) => 0.0,
            Some(Equal) => 0.5,
            Some(Greater) => 1.0,
            None => unreachable!(),
        }
    }

    #[inline]
    fn modes(&self) -> Vec<u8> {
        use std::cmp::Ordering::*;
        match self.p.partial_cmp(&self.q) {
            Some(Less) => vec![0],
            Some(Equal) => vec![0, 1],
            Some(Greater) => vec![1],
            None => unreachable!(),
        }
    }

    #[inline]
    fn entropy(&self) -> f64 {
        -self.q * self.q.ln() - self.p * self.p.ln()
    }

    #[inline]
    fn cdf(&self, x: u8) -> f64 {
        if x == 0 { self.q } else { 1.0 }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> u8 {
        should!(0.0 <= p && p <= 1.0);
        if p <= self.q { 0 } else { 1 }
    }

    #[inline]
    fn pdf(&self, x: u8) -> f64 {
        if x == 0 { self.q } else if x == 1 { self.p } else { 0.0 }
    }

    #[inline]
    fn sample<G>(&self, generator: &mut G) -> u8 where G: Generator {
        if generator.next::<f64>() < self.q { 0 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        (failure $q:expr) => (Bernoulli::new_failprob($q));
        ($p:expr) => (Bernoulli::new($p));
    );

    #[test]
    fn mean() {
        assert_eq!(new!(0.5).mean(), 0.5);
    }

    #[test]
    fn var() {
        assert_eq!(new!(0.25).var(), 0.1875);
    }

    #[test]
    fn sd() {
        assert_eq!(new!(0.5).sd(), 0.5);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(0.5).skewness(), 0.0);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(0.5).kurtosis(), -2.0);
    }

    #[test]
    fn median() {
        assert_eq!(new!(0.25).median(), 0.0);
        assert_eq!(new!(0.5).median(), 0.5);
        assert_eq!(new!(0.75).median(), 1.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(0.25).modes(), vec![0]);
        assert_eq!(new!(0.5).modes(), vec![0, 1]);
        assert_eq!(new!(0.75).modes(), vec![1]);
    }

    #[test]
    fn entropy() {
        let bernoullies = vec![new!(0.25), new!(0.5), new!(0.75)];
        assert::close(&bernoullies.iter().map(|d| d.entropy()).collect::<Vec<_>>(),
                      &vec![0.5623351446188083, 0.6931471805599453, 0.5623351446188083], 1e-16);
    }

    #[test]
    fn pdf() {
        let bernoulli = new!(0.25);
        assert_eq!(&(0..3).map(|x| bernoulli.pdf(x)).collect::<Vec<_>>(), &[0.75, 0.25, 0.0]);
    }

    #[test]
    fn cdf() {
        let bernoulli = new!(0.25);
        assert_eq!(&(0..3).map(|x| bernoulli.cdf(x)).collect::<Vec<_>>(), &[0.75, 1.0, 1.0]);
    }

    #[test]
    fn inv_cdf() {
        let bernoulli = new!(0.25);
        let p = vec![0.0, 0.25, 0.5, 0.75, 0.75000000001, 1.0];
        let x = vec![0, 0, 0, 0, 1, 1];
        assert_eq!(&p.iter().map(|&p| bernoulli.inv_cdf(p)).collect::<Vec<_>>(), &x);
    }

    #[test]
    fn sample() {
        assert!(Sampler(&new!(0.25), &mut generator()).take(100).fold(0, |a, b| a + b) <= 100);
    }
}
