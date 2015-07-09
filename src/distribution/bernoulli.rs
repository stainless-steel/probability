use distribution;
use random;

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
    pub fn with_failure(q: f64) -> Bernoulli {
        should!(q > 0.0 && q < 1.0);
        Bernoulli { p: 1.0 - q, q: q, pq: (1.0 - q) * q }
    }

    /// Return the success probability.
    #[inline(always)]
    pub fn p(&self) -> f64 { self.p }

    /// Return the failure probability.
    #[inline(always)]
    pub fn q(&self) -> f64 { self.q }
}

impl distribution::Distribution for Bernoulli {
    type Value = u8;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x < 1.0 {
            self.q
        } else {
            1.0
        }
    }
}

impl distribution::Discrete for Bernoulli {
    #[inline]
    fn pmf(&self, x: u8) -> f64 {
        if x == 0 {
            self.q
        } else if x == 1 {
            self.p
        } else {
            0.0
        }
    }
}

impl distribution::Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        -self.q * self.q.ln() - self.p * self.p.ln()
    }
}

impl distribution::Inverse for Bernoulli {
    #[inline]
    fn inv_cdf(&self, p: f64) -> u8 {
        should!(0.0 <= p && p <= 1.0);
        if p <= self.q { 0 } else { 1 }
    }
}

impl distribution::Kurtosis for Bernoulli {
    #[inline]
    fn kurtosis(&self) -> f64 {
        (1.0 - 6.0 * self.pq) / (self.pq)
    }
}

impl distribution::Mean for Bernoulli {
    #[inline]
    fn mean(&self) -> f64 { self.p }
}

impl distribution::Median for Bernoulli {
    fn median(&self) -> f64 {
        use std::cmp::Ordering::*;
        match self.p.partial_cmp(&self.q) {
            Some(Less) => 0.0,
            Some(Equal) => 0.5,
            Some(Greater) => 1.0,
            None => unreachable!(),
        }
    }
}

impl distribution::Modes for Bernoulli {
    fn modes(&self) -> Vec<u8> {
        use std::cmp::Ordering::*;
        match self.p.partial_cmp(&self.q) {
            Some(Less) => vec![0],
            Some(Equal) => vec![0, 1],
            Some(Greater) => vec![1],
            None => unreachable!(),
        }
    }
}

impl distribution::Sample for Bernoulli {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> u8 where S: random::Source {
        if source.read::<f64>() < self.q { 0 } else { 1 }
    }
}

impl distribution::Skewness for Bernoulli {
    #[inline]
    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p) / (self.pq).sqrt()
    }
}

impl distribution::Variance for Bernoulli {
    #[inline]
    fn variance(&self) -> f64 { self.pq }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        (failure $q:expr) => (Bernoulli::with_failure($q));
        ($p:expr) => (Bernoulli::new($p));
    );

    #[test]
    fn cdf() {
        let d = new!(0.25);
        let x = vec![-0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 1.1];
        let p = vec![0.0, 0.75, 0.75, 0.75, 0.75, 1.0, 1.0];
        assert_eq!(&x.iter().map(|&x| d.cdf(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn pmf() {
        let d = new!(0.25);
        assert_eq!(&(0..3).map(|x| d.pmf(x)).collect::<Vec<_>>(), &[0.75, 0.25, 0.0]);
    }

    #[test]
    fn entropy() {
        let d = vec![new!(0.25), new!(0.5), new!(0.75)];
        assert::close(&d.iter().map(|d| d.entropy()).collect::<Vec<_>>(),
                      &vec![0.5623351446188083, 0.6931471805599453, 0.5623351446188083], 1e-16);
    }

    #[test]
    fn inv_cdf() {
        let d = new!(0.25);
        let p = vec![0.0, 0.25, 0.5, 0.75, 0.75000000001, 1.0];
        let x = vec![0, 0, 0, 0, 1, 1];
        assert_eq!(&p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>(), &x);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(0.5).kurtosis(), -2.0);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(0.5).mean(), 0.5);
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
    fn sample() {
        assert!(Independent(&new!(0.25), &mut random::default()).take(100)
                                                                .fold(0, |a, b| a + b) <= 100);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(0.5).skewness(), 0.0);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(0.25).variance(), 0.1875);
    }
}
