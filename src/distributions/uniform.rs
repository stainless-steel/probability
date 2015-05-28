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
    /// Create a uniform distribution on the interval `[a, b]`.
    ///
    /// # Panics
    ///
    /// Panics if `a >= b`.
    #[inline]
    pub fn new(a: f64, b: f64) -> Uniform {
        assert!(a < b, "Uniform::new() called with a >= b");
        Uniform { a: a, b: b }
    }
}

impl Distribution for Uniform {
    type Item = f64;

    fn mean(&self) -> f64 { (self.a + self.b) / 2.0 }

    fn var(&self) -> f64 { (self.b - self.a).powi(2) / 12.0 }

    fn sd(&self) -> f64 { self.var().sqrt() }

    fn median(&self) -> f64 { self.mean() }

    fn modes(&self) -> Vec<f64> { Vec::new() }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { -1.2 }

    fn entropy(&self) -> f64 { (self.b - self.a).ln() }

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

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> f64 {
        self.a + (self.b - self.a) * generator.gen::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use {Distribution, Sampler};
    use distributions::Uniform;

    #[test]
    #[should_panic]
    #[allow(unused_variables)]
    fn invalid_support() {
        let uniform = Uniform::new(2.0, -1.0);
    }

    #[test]
    fn mean() {
        let d = Uniform::new(0.0, 2.0);
        assert_eq!(d.mean(), 1.0);
    }

    #[test]
    fn var() {
        let d = Uniform::new(0.0, 12.0);
        assert_eq!(d.var(), 12.0);
    }

    #[test]
    fn sd() {
        let d = Uniform::new(0.0, 12.0);
        assert_eq!(d.sd(), (12 as f64).sqrt());
    }

    #[test]
    fn median() {
        let d = Uniform::new(0.0, 2.0);
        assert_eq!(d.median(), 1.0);
    }

    #[test]
    fn modes() {
        let d = Uniform::new(0.0, 2.0);
        assert_eq!(d.modes(), Vec::<f64>::new());
    }

    #[test]
    fn skewness() {
        let d = Uniform::new(0.0, 2.0);
        assert_eq!(d.skewness(), 0.0);
    }

    #[test]
    fn kurtosis() {
        let d = Uniform::new(0.0, 2.0);
        assert_eq!(d.kurtosis(), -1.2);
    }

    #[test]
    fn entropy() {
        use std::f64::consts::E;
        let d = Uniform::new(0.0, E);
        assert_eq!(d.entropy(), 1.0);
    }

    #[test]
    fn pdf() {
        let uniform = Uniform::new(-1.0, 1.0);
        let x = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let p = vec![0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0];

        assert_eq!(x.iter().map(|&x| uniform.pdf(x)).collect::<Vec<_>>(), p);
    }

    #[test]
    fn cdf() {
        let uniform = Uniform::new(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        assert_eq!(x.iter().map(|&x| uniform.cdf(x)).collect::<Vec<_>>(), p);
    }

    #[test]
    fn inv_cdf() {
        let uniform = Uniform::new(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        assert_eq!(p.iter().map(|&p| uniform.inv_cdf(p)).collect::<Vec<_>>(), x);
    }

    #[test]
    fn sample() {
        let mut generator = ::generator();
        let uniform = Uniform::new(7.0, 42.0);

        for x in Sampler(&uniform, &mut generator).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }
}
