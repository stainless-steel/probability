use distribution::{self, Gaussian};
use source::Source;

/// A lognormal distribution.
#[derive(Clone, Copy, Debug)]
pub struct Lognormal {
    mu: f64,
    sigma: f64,
    gaussian: Gaussian,
}

impl Lognormal {
    /// Create a lognormal distribution with location `mu` and scale `sigma`.
    ///
    /// It should hold that `sigma > 0`.
    #[inline]
    pub fn new(mu: f64, sigma: f64) -> Self {
        should!(sigma > 0.0);
        Lognormal {
            mu,
            sigma,
            gaussian: Gaussian::new(mu, sigma),
        }
    }

    /// Return the location parameter.
    #[inline(always)]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Return the scale parameter.
    #[inline(always)]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Default for Lognormal {
    #[inline]
    fn default() -> Self {
        Lognormal::new(0.0, 1.0)
    }
}

impl distribution::Continuous for Lognormal {
    fn density(&self, x: f64) -> f64 {
        use std::f64::consts::PI;
        if x <= 0.0 {
            0.0
        } else {
            let &Lognormal { mu, sigma, .. } = self;
            (-(x.ln() - mu).powi(2) / (2.0 * sigma * sigma)).exp() / (x * sigma * (2.0 * PI).sqrt())
        }
    }
}

impl distribution::Distribution for Lognormal {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            self.gaussian.distribution(x.ln())
        }
    }
}

impl distribution::Entropy for Lognormal {
    #[inline]
    fn entropy(&self) -> f64 {
        use std::f64::consts::PI;
        (self.sigma * (self.mu + 0.5).exp() * (2.0 * PI).sqrt()).ln()
    }
}

impl distribution::Inverse for Lognormal {
    fn inverse(&self, p: f64) -> f64 {
        self.gaussian.inverse(p).exp()
    }
}

impl distribution::Kurtosis for Lognormal {
    #[inline]
    fn kurtosis(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0
    }
}

impl distribution::Mean for Lognormal {
    #[inline]
    fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }
}

impl distribution::Median for Lognormal {
    #[inline]
    fn median(&self) -> f64 {
        self.mu.exp()
    }
}

impl distribution::Modes for Lognormal {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![(self.mu - self.sigma * self.sigma).exp()]
    }
}

impl distribution::Sample for Lognormal {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        self.gaussian.sample(source).exp()
    }
}

impl distribution::Skewness for Lognormal {
    #[inline]
    fn skewness(&self) -> f64 {
        let es2 = (self.sigma * self.sigma).exp();
        (es2 - 1.0).sqrt() * (2.0 + es2)
    }
}

impl distribution::Variance for Lognormal {
    #[inline]
    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (s2.exp() - 1.0) * (2.0 * self.mu + s2).exp()
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($mu:expr, $sigma:expr) => (Lognormal::new($mu, $sigma));
    );

    #[test]
    fn density() {
        let d = new!(1.0, 2.0);
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let p = vec![
            0.0000000000000000e+00,
            2.7879404629273086e-01,
            1.7603266338214976e-01,
            1.2723305581441105e-01,
            9.8568580344013113e-02,
            7.9718599555316239e-02,
            6.6409606924506773e-02,
            5.6538422820400766e-02,
            4.8946227003151078e-02,
            4.2941143217487855e-02,
            3.8084403129689012e-02,
        ];

        assert::close(
            &x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(),
            &p,
            1e-15,
        );
    }

    #[test]
    fn distribution() {
        let d = new!(1.0, 2.0);
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let p = vec![
            0.0000000000000000e+00,
            1.9861641975736130e-01,
            3.0853753872598694e-01,
            3.8313116661630492e-01,
            4.3903100974768944e-01,
            4.8330729072740009e-01,
            5.1966233849751675e-01,
            5.5028502097208276e-01,
            5.7657814823924480e-01,
            5.9949442394950303e-01,
            6.1970989457732906e-01,
        ];

        assert::close(
            &x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(),
            &p,
            1e-15,
        );
    }

    #[test]
    fn entropy() {
        use std::f64::consts::PI;
        assert_eq!(new!(-0.5, 1.0 / (2.0 * PI).sqrt()).entropy(), 0.0);
    }

    #[test]
    fn inverse() {
        use std::f64::INFINITY;
        let d = new!(1.0, 2.0);
        let p = vec![
            0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
            0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
        ];
        let x = vec![
            0.0000000000000000e+00,
            1.0129611155505908e-01,
            2.0948500212405705e-01,
            3.4202659595680435e-01,
            5.0497696371871126e-01,
            7.0540759070071157e-01,
            9.5237060839269883e-01,
            1.2577935903399797e+00,
            1.6377212497125082e+00,
            2.1142017250556107e+00,
            2.7182818284590451e+00,
            3.4949626666945868e+00,
            4.5117910634839467e+00,
            5.8746173900706555e+00,
            7.7585931714136498e+00,
            1.0474874663016855e+01,
            1.4632461735515125e+01,
            2.1603747153814467e+01,
            3.5272482631261830e+01,
            7.2945110977081981e+01,
            INFINITY,
        ];

        assert::close(
            &p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(),
            &x,
            1e-12,
        );
    }

    #[test]
    fn kurtosis() {
        assert::close(new!(0.0, 1.0).kurtosis(), 1.1093639217631153e+02, 1e-15);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(-2.0, 2.0).mean(), 1.0);
    }

    #[test]
    fn median() {
        assert_eq!(new!(0.0, 1.0).median(), 1.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(1.0, 1.0).modes(), vec![1.0]);
    }

    #[test]
    fn skewness() {
        assert!(4.0 - new!(0.0, 2f64.ln().sqrt()).skewness() < 1e-10);
    }

    #[test]
    fn variance() {
        assert!(2.0 - new!(0.0, 2f64.ln().sqrt()).variance() < 1e-10);
    }

    #[test]
    fn deviation() {
        assert!(2f64.sqrt() - new!(0.0, 2f64.ln().sqrt()).variance() < 1e-10);
    }
}
