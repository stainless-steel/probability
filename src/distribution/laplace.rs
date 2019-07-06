use distribution;
use distribution::Inverse;
use source::Source;

/// A Laplace distribution.
#[derive(Clone, Copy, Debug)]
pub struct Laplace {
    mu: f64,
    b: f64,
}

impl Laplace {
    /// Create a Laplace distribution with location `mu` and scale `b`.
    ///
    /// It should hold that `b > 0`.
    #[inline]
    pub fn new(mu: f64, b: f64) -> Self {
        should!(b > 0.0);
        Laplace { mu: mu, b: b }
    }

    // Return the location parameter
    #[inline(always)]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    // Return the scale parameter
    #[inline(always)]
    pub fn b(&self) -> f64 {
        self.b
    }
}

impl distribution::Continuous for Laplace {
    #[inline]
    fn density(&self, x: f64) -> f64 {
        self.b.recip() * 0.5 * (-(x - self.mu).abs() / self.b).exp()
    }
}

impl distribution::Distribution for Laplace {
    type Value = f64;

    #[inline]
    fn distribution(&self, x: f64) -> f64 {
        if x <= self.mu {
            0.5 * ((x - self.mu) / self.b).exp()
        } else {
            1.0 - 0.5 * (-(x - self.mu) / self.b).exp()
        }
    }
}

impl distribution::Entropy for Laplace {
    #[inline]
    fn entropy(&self) -> f64 {
        (std::f64::consts::E * 2.0 * self.b).ln()
    }
}

impl distribution::Inverse for Laplace {
    #[inline]
    fn inverse(&self, p: f64) -> f64 {
        should!(0.0 <= p && p <= 1.0);
        if p > 0.5 {
            if p == 1.0 {
                return std::f64::INFINITY;
            }
            self.mu - self.b * (2.0 - 2.0 * p).ln()
        } else {
            if p == 0.0 {
                return std::f64::NEG_INFINITY;
            }
            self.mu + self.b * (2.0 * p).ln()
        }
    }
}

impl distribution::Kurtosis for Laplace {
    #[inline]
    fn kurtosis(&self) -> f64 {
        3.0
    }
}

impl distribution::Mean for Laplace {
    #[inline]
    fn mean(&self) -> f64 {
        self.mu
    }
}

impl distribution::Median for Laplace {
    #[inline]
    fn median(&self) -> f64 {
        self.mu
    }
}

impl distribution::Modes for Laplace {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![self.mu]
    }
}

impl distribution::Sample for Laplace {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        self.inverse(source.read::<f64>())
    }
}

impl distribution::Skewness for Laplace {
    #[inline]
    fn skewness(&self) -> f64 {
        0.0
    }
}

impl distribution::Variance for Laplace {
    #[inline]
    fn variance(&self) -> f64 {
        2.0 * self.b.powi(2)
    }

    #[inline]
    fn deviation(&self) -> f64 {
        f64::sqrt(2.0) * self.b
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($mu:expr, $b:expr) => (Laplace::new($mu, $b));
    );

    #[test]
    fn density() {
        let d = new!(2.0, 8.0);
        let x = vec![-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 12.0];
        let p = vec![
            0.042955579924435765,
            0.048675048941962805,
            0.05181431988627502,
            0.055156056411537216,
            0.05871331642584224,
            0.0625,
            0.05871331642584224,
            0.055156056411537216,
            0.048675048941962805,
            0.03790816623203959,
            0.01790654980376188,
        ];

        assert::close(
            &x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(),
            &p,
            1e-15,
        );
    }

    #[test]
    fn distribution() {
        let d = new!(2.0, 8.0);
        let x = vec![
            -1.0, 0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
        ];
        let p = vec![
            0.3436446393954861,
            0.38940039153570244,
            0.3898874463709755,
            0.39184176532872866,
            0.39429844549053833,
            0.39677052798551266,
            0.4017612868445304,
            0.4145145590902002,
            0.4412484512922977,
            0.4697065314067379,
            0.5,
            0.5587515487077023,
            0.6105996084642975,
        ];

        assert::close(
            &x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(),
            &p,
            1e-15,
        );
    }

    #[test]
    fn entropy() {
        use std::f64::consts::E;
        assert_eq!(new!(2.0, 1.0).entropy(), (2.0 * 1.0 * E).ln());
    }

    #[test]
    fn inverse() {
        let d = new!(2.0, 3.0);
        let x = vec![
            std::f64::NEG_INFINITY,
            -2.8283137373023006,
            -0.07944154167983575,
            2.0,
            4.079441541679836,
            6.8283137373023015,
            std::f64::INFINITY,
        ];
        let p = vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.00];

        assert::close(
            &p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(),
            &x,
            1e-14,
        );
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(2.0, 9.0).kurtosis(), 3.0);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(2.0, 1.0).mean(), 2.0);
    }

    #[test]
    fn median() {
        assert_eq!(new!(2.0, 1.0).median(), 2.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(2.0, 1.0).modes(), vec![2.0]);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(2.0, 1.0).skewness(), 0.0);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(2.0, 3.0).variance(), 18.0);
    }

    #[test]
    fn deviation() {
        assert::close(new!(2.0, 3.0).deviation(), 4.242640687119286, 1e-7);
    }
}
