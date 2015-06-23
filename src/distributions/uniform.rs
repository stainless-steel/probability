use {Distribution, Generator};

/// A continuous uniform distribution.
#[derive(Clone, Copy)]
pub struct Uniform {
    /// The left endpoint of the support.
    pub a: f64,
    /// The right endpoint of the support.
    pub b: f64,
}

impl Uniform {
    /// Create a uniform distribution on interval `[a, b]`.
    ///
    /// It should hold that `a < b`.
    #[inline]
    pub fn new(a: f64, b: f64) -> Uniform {
        should!(a < b);
        Uniform { a: a, b: b }
    }
}

impl Distribution for Uniform {
    type Value = f64;

    #[inline]
    fn mean(&self) -> f64 { (self.a + self.b) / 2.0 }

    #[inline]
    fn var(&self) -> f64 { (self.b - self.a).powi(2) / 12.0 }

    #[inline]
    fn skewness(&self) -> f64 { 0.0 }

    #[inline]
    fn kurtosis(&self) -> f64 { -1.2 }

    #[inline]
    fn median(&self) -> f64 { self.mean() }

    #[inline]
    fn modes(&self) -> Vec<f64> { Vec::new() }

    #[inline]
    fn entropy(&self) -> f64 { (self.b - self.a).ln() }

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if self.b <= x {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> f64 {
        should!(0.0 <= p && p <= 1.0);
        self.a + (self.b - self.a) * p
    }

    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (self.b - self.a)
        }
    }

    #[inline(always)]
    fn sample<G: Generator>(&self, generator: &mut G) -> f64 {
        self.a + (self.b - self.a) * generator.next::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use {Distribution, Sampler};
    use distributions::Uniform;

    macro_rules! new(
        ($a:expr, $b:expr) => (Uniform::new($a, $b));
    );

    #[test]
    fn mean() {
        assert_eq!(new!(0.0, 2.0).mean(), 1.0);
    }

    #[test]
    fn var() {
        assert_eq!(new!(0.0, 12.0).var(), 12.0);
    }

    #[test]
    fn sd() {
        assert_eq!(new!(0.0, 12.0).sd(), 12f64.sqrt());
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(0.0, 2.0).skewness(), 0.0);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(0.0, 2.0).kurtosis(), -1.2);
    }

    #[test]
    fn median() {
        assert_eq!(new!(0.0, 2.0).median(), 1.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(0.0, 2.0).modes(), Vec::<f64>::new());
    }

    #[test]
    fn entropy() {
        use std::f64::consts::E;
        assert_eq!(new!(0.0, E).entropy(), 1.0);
    }

    #[test]
    fn pdf() {
        let uniform = new!(-1.0, 1.0);
        let x = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let p = vec![0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0];

        assert_eq!(&x.iter().map(|&x| uniform.pdf(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn cdf() {
        let uniform = new!(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        assert_eq!(&x.iter().map(|&x| uniform.cdf(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn inv_cdf() {
        let uniform = new!(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        assert_eq!(&p.iter().map(|&p| uniform.inv_cdf(p)).collect::<Vec<_>>(), &x);
    }

    #[test]
    fn sample() {
        for x in Sampler(&new!(7.0, 42.0), &mut ::generator()).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }
}
