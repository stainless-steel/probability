use {Distribution, Generator};

/// A Bernoulli distribution.
#[derive(Clone, Copy)]
pub struct Bernoulli {
    /// The probability of success.
    pub p: f64,
    /// The probability of failure.
    pub q: f64,
    // The variance
    pq: f64,
}

impl Bernoulli {
    /// Create a Bernoulli distribution with success probability `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p < 0` or `p > 1`.
    #[inline]
    pub fn new(p: f64) -> Bernoulli {
        debug_assert!(0. < p && p < 1., "Bernoulli::new() is called with p < 0 or p > 1");
        let q = 1. - p;
        Bernoulli { p: p, q: q, pq: p * q }
    }

    /// Create a Bernoulli distribution with failure probability `q`.
    ///
    /// # Panics
    ///
    /// Panics if `q < 0` or `q > 1`.
    #[inline]
    pub fn new_failprob(q: f64) -> Bernoulli {
        debug_assert!(0. < q && q < 1., "Bernoulli::new_failprob() is called with q < 0 or q > 1");
        let p = 1. - q;
        Bernoulli { p: p, q: q, pq: p * q }
    }
    
}

impl Distribution for Bernoulli {
    type Value = u8;

    #[inline]
    fn mean(&self) -> f64 { self.p }

    #[inline]
    fn var(&self) -> f64 { self.pq }

    #[inline]
    fn skewness(&self) -> f64 { (1. - 2. * self.p) / (self.pq).sqrt() }

    #[inline]
    fn kurtosis(&self) -> f64 { (1. - 6. * self.pq) / (self.pq) }

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
    fn modes(&self) -> Vec<Self::Value> {
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
    fn cdf(&self, x: Self::Value) -> f64 {
        if x == 0 { self.q }
        else { 1. }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> Self::Value {
        debug_assert!(0.0 <= p && p <= 1.0, "inv_cdf is called with p outside of [0, 1]");
        if p <= self.q { 0 }
        else { 1 }
    }

    #[inline]
    fn pdf(&self, x: Self::Value) -> f64 {
        if x == 0 { self.q }
        else if x == 1 { self.p }
        else { 0.0 }
    }

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> Self::Value {
        if generator.gen::<f64>() < self.q { 0 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use assert;

    use {Distribution, Sampler};
    use distributions::Bernoulli;

    #[test]
    #[should_panic]
    #[allow(unused_variables)]
    fn invalid_succprob() {
        let _ = Bernoulli::new(2.0);
        let _ = Bernoulli::new(-0.5);
    }

    #[test]
    #[should_panic]
    #[allow(unused_variables)]
    fn invalid_failprob() {
        let _ = Bernoulli::new_failprob(2.0);
        let _ = Bernoulli::new_failprob(-0.5);
    }

    #[test]
    fn new_failprob() {
        let _ = Bernoulli::new_failprob(1e-24);
    }

    #[test]
    fn mean() {
        let d = Bernoulli::new(0.5);
        assert_eq!(d.mean(), 0.5);
    }

    #[test]
    fn var() {
        let d = Bernoulli::new(0.25);
        assert_eq!(d.var(), 0.1875);
    }

    #[test]
    fn sd() {
        let d = Bernoulli::new(0.5);
        assert_eq!(d.sd(), 0.5);
    }

    #[test]
    fn skewness() {
        let d = Bernoulli::new(0.5);
        assert_eq!(d.skewness(), 0.0);
    }

    #[test]
    fn kurtosis() {
        let d = Bernoulli::new(0.5);
        assert_eq!(d.kurtosis(), -2.0);
    }

    #[test]
    fn median() {
        let d1 = Bernoulli::new(0.25);
        let d2 = Bernoulli::new(0.5);
        let d3 = Bernoulli::new(0.75);
        assert_eq!(d1.median(), 0.0);
        assert_eq!(d2.median(), 0.5);
        assert_eq!(d3.median(), 1.0);
    }

    #[test]
    fn modes() {
        let d1 = Bernoulli::new(0.25);
        let d2 = Bernoulli::new(0.5);
        let d3 = Bernoulli::new(0.75);
        assert_eq!(d1.modes(), vec![0]);
        assert_eq!(d2.modes(), vec![0, 1]);
        assert_eq!(d3.modes(), vec![1]);
    }

    #[test]
    fn entropy() {
        let dists = vec![
            Bernoulli::new(0.25),
            Bernoulli::new(0.5),
            Bernoulli::new(0.75),
        ];
        assert::within(&dists.iter().map(|d| d.entropy()).collect::<Vec<_>>(),
                       &vec![0.5623351446188083, 0.6931471805599453, 0.5623351446188083], 1e-16);
    }

    #[test]
    fn pdf() {
        let bernoulli = Bernoulli::new(0.25);
        let x = 0..3;
        let p = vec![0.75, 0.25, 0.0];

        assert::equal(&x.map(|x| bernoulli.pdf(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn cdf() {
        let bernoulli = Bernoulli::new(0.25);
        let x = 0..3;
        let p = vec![0.75, 1., 1.];

        assert::equal(&x.map(|x| bernoulli.cdf(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn inv_cdf() {
        let bernoulli = Bernoulli::new(0.25);
        let p = vec![0.0, 0.25, 0.5, 0.75, 0.75000000001, 1.0];
        let x = vec![0, 0, 0, 0, 1, 1];

        assert::equal(&p.iter().map(|&p| bernoulli.inv_cdf(p)).collect::<Vec<_>>(), &x);
    }

    #[test]
    fn sample() {
        let mut generator = ::generator();
        let bernoulli = Bernoulli::new(0.25);

        let sum = Sampler(&bernoulli, &mut generator)
            .take(100)
            .fold(0, |a, b| a + b);

        assert!(sum <= 100);
    }
}
