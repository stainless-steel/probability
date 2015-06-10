use {Distribution, Generator};

/// A discrete Bernoulli distribution.
#[derive(Clone, Copy)]
pub struct Bernoulli {
    /// The success probability.
    pub p: f64,
    // The probability of failure.
    q: f64,
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
        Bernoulli { p: p, q: 1.0 - p }
    }
}

impl Distribution for Bernoulli {
    type Value = i32;

    #[inline]
    fn mean(&self) -> f64 { self.p }

    #[inline]
    fn var(&self) -> f64 { self.p * self.q }

    #[inline]
    fn skewness(&self) -> f64 { (1. - 2. * self.p) / (self.p * self.q).sqrt() }

    #[inline]
    fn kurtosis(&self) -> f64 { (1. - 6. * self.p * self.q) / (self.p * self.q) }

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
    fn modes(&self) -> Vec<i32> {
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
    fn cdf(&self, x: i32) -> f64 {
        if x < 0 { 0. }
        else if x < 1 { self.q }
        else { 1. }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> i32 {
        debug_assert!(0.0 <= p && p <= 1.0, "inv_cdf is called with p outside of [0, 1]");
        if p <= self.q { 0 }
        else { 1 }
    }

    #[inline]
    fn pdf(&self, x: i32) -> f64 {
        if x == 0 { self.q }
        else if x == 1 { self.p }
        else { 0.0 }
    }

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> i32 {
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
        let x = -1..3;
        let p = vec![0.0, 0.75, 0.25, 0.0];

        assert::equal(&x.map(|x| bernoulli.pdf(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn cdf() {
        let bernoulli = Bernoulli::new(0.25);
        let x = -1..3;
        let p = vec![0., 0.75, 1., 1.];

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

        assert!(0 <= sum && sum <= 100);
    }
}