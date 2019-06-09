use distribution;
use source::Source;

/// A PERT distribution.
#[derive(Clone, Copy, Debug)]
pub struct Pert {
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    ln_beta: f64,
}

impl Pert {
    /// Create a PERT distribution with parameters `a`, `b`, and `c`.
    ///
    /// It should hold that `a < b < c`.
    #[inline]
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        use special::Beta as SpecialBeta;
        should!(a < b && b < c);
        let alpha = (4.0 * b + c - 5.0 * a) / (c - a);
        let beta = (5.0 * c - a - 4.0 * b) / (c - a);
        Pert {
            a: a,
            b: b,
            c: c,
            alpha: alpha,
            beta: beta,
            ln_beta: alpha.ln_beta(beta),
        }
    }

    /// Return the first parameter.
    #[inline(always)]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Return the second parameter.
    #[inline(always)]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Return the third parameter.
    #[inline(always)]
    pub fn c(&self) -> f64 {
        self.c
    }

    /// Return the first shape parameter of the corresponding Beta distribution.
    #[inline(always)]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the second shape parameter of the corresponding Beta distribution.
    #[inline(always)]
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl distribution::Continuous for Pert {
    fn density(&self, x: f64) -> f64 {
        if x < self.a || x > self.c {
            0.0
        } else {
            let scale = self.c - self.a;
            let x = (x - self.a) / scale;
            ((self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (-x).ln_1p() - self.ln_beta).exp()
                / scale
        }
    }
}

impl distribution::Distribution for Pert {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        use special::Beta;
        if x <= self.a {
            0.0
        } else if x >= self.c {
            1.0
        } else {
            ((x - self.a) / (self.c - self.a)).inc_beta(self.alpha, self.beta, self.ln_beta)
        }
    }
}

impl distribution::Entropy for Pert {
    fn entropy(&self) -> f64 {
        use special::Gamma;
        let sum = self.alpha + self.beta;
        (self.c - self.a).ln() + self.ln_beta
            - (self.alpha - 1.0) * self.alpha.digamma()
            - (self.beta - 1.0) * self.beta.digamma()
            + (sum - 2.0) * sum.digamma()
    }
}

impl distribution::Inverse for Pert {
    #[inline]
    fn inverse(&self, p: f64) -> f64 {
        use special::Beta;
        should!(0.0 <= p && p <= 1.0);
        self.a + (self.c - self.a) * p.inv_inc_beta(self.alpha, self.beta, self.ln_beta)
    }
}

impl distribution::Kurtosis for Pert {
    fn kurtosis(&self) -> f64 {
        let sum = self.alpha + self.beta;
        let delta = self.alpha - self.beta;
        let product = self.alpha * self.beta;
        6.0 * (delta * delta * (sum + 1.0) - product * (sum + 2.0))
            / (product * (sum + 2.0) * (sum + 3.0))
    }
}

impl distribution::Mean for Pert {
    #[inline]
    fn mean(&self) -> f64 {
        (self.a + self.b * 4.0 + self.c) / 6.0
    }
}

impl distribution::Median for Pert {
    fn median(&self) -> f64 {
        use distribution::Inverse;
        self.inverse(0.5)
    }
}

impl distribution::Modes for Pert {
    fn modes(&self) -> Vec<f64> {
        vec![self.b]
    }
}

impl distribution::Sample for Pert {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        use distribution::gamma;
        let x = gamma::sample(self.alpha, source);
        let y = gamma::sample(self.beta, source);
        self.a + (self.c - self.a) * x / (x + y)
    }
}

impl distribution::Skewness for Pert {
    fn skewness(&self) -> f64 {
        let sum = self.alpha + self.beta;
        2.0 * (self.beta - self.alpha) * (sum + 1.0).sqrt()
            / ((sum + 2.0) * (self.alpha * self.beta).sqrt())
    }
}

impl distribution::Variance for Pert {
    fn variance(&self) -> f64 {
        use distribution::Mean;
        (self.mean() - self.a) * (self.c - self.mean()) / 7.0
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($a:expr, $b:expr, $c:expr) => (Pert::new($a, $b, $c));
    );

    #[test]
    fn density() {
        let d = new!(-1.0, 0.5, 2.0);
        let beta = Beta::new(3.0, 3.0, -1.0, 2.0);
        let x = vec![-1.15, -1.0, -0.85, -0.5, 0.0, 0.5, 1.0, 1.5, 1.85, 2.0];
        let p = vec![
            0.0,
            0.0,
            0.022562499999999996,
            0.19290123456790118,
            0.4938271604938269,
            0.6249999999999999,
            0.49382716049382713,
            0.1929012345679011,
            0.022562499999999933,
            0.0,
        ];
        assert::close(
            &x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(),
            &x.iter().map(|&x| beta.density(x)).collect::<Vec<_>>(),
            1e-14,
        );
        assert::close(
            &x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(),
            &p,
            1e-14,
        );
    }

    #[test]
    fn distribution() {
        let d = new!(-1.0, 0.5, 2.0);
        let beta = Beta::new(3.0, 3.0, -1.0, 2.0);
        let x = vec![-1.15, -1.0, -0.85, -0.5, 0.0, 0.5, 1.0, 1.5, 1.85, 2.0];
        let p = vec![
            0.0,
            0.0,
            0.001158125,
            0.03549382716049382,
            0.20987654320987656,
            0.5,
            0.7901234567901234,
            0.9645061728395061,
            0.998841875,
            1.0,
        ];
        assert::close(
            &x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(),
            &x.iter().map(|&x| beta.distribution(x)).collect::<Vec<_>>(),
            1e-14,
        );
        assert::close(
            &x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(),
            &p,
            1e-14,
        );
    }

    #[test]
    fn entropy() {
        use std::f64::consts::E;
        let d = vec![
            new!(0.0, 0.5, 1.0),
            new!(0.0, 0.5, E),
            new!(0.0, 0.3, 1.0),
            new!(-1.0, 1.0, 2.0),
        ];
        assert::close(
            &d.iter().map(|d| d.entropy()).collect::<Vec<_>>(),
            &d.iter()
                .map(|d| Beta::new(d.alpha(), d.beta(), d.a(), d.c()).entropy())
                .collect::<Vec<_>>(),
            1e-15,
        );
    }

    #[test]
    fn inverse() {
        let d = new!(-1.0, 0.5, 2.0);
        let p = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let x = vec![
            -1.0,
            -0.020206186475766774,
            0.33876229245942,
            0.6612377075405802,
            1.0202061864757672,
            2.0,
        ];
        assert::close(
            &p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(),
            &x,
            1e-14,
        );
    }

    #[test]
    fn kurtosis() {
        assert::close(new!(0.0, 0.5, 1.0).kurtosis(), -2.0 / 3.0, 1e-14);
    }

    #[test]
    fn mean() {
        assert::close(new!(0.0, 0.5, 1.0).mean(), 0.5, 1e-14);
        assert::close(
            new!(-1.0, 1.5, 2.0).mean(),
            (1.5 * 4.0 - 1.0 + 2.0) / 6.0,
            1e-14,
        );
        assert::close(
            Beta::new(3.0, 3.0, -1.0, 2.0).mean(),
            (0.5 * 4.0 - 1.0 + 2.0) / 6.0,
            1e-14,
        );
    }

    #[test]
    fn median() {
        assert::close(new!(0.0, 0.5, 1.0).median(), 0.5, 1e-14);
        assert::close(new!(0.0, 0.3, 1.0).median(), 0.3509994849491181, 1e-14);
    }

    #[test]
    fn modes() {
        assert::close(new!(-1.0, 0.5, 2.0).modes(), vec![0.5], 1e-14);
    }

    #[test]
    fn sample() {
        for x in Independent(&new!(7.0, 20.0, 42.0), &mut source::default()).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }

    #[test]
    fn skewness() {
        assert::close(new!(0.0, 0.5, 1.0).skewness(), 0.0, 1e-14);
        assert::close(new!(-1.0, 0.2, 2.0).skewness(), 0.17797249266332246, 1e-14);
        assert::close(new!(-1.0, 0.8, 2.0).skewness(), -0.17797249266332246, 1e-14);
    }

    #[test]
    fn variance() {
        assert::close(new!(0.0, 0.5, 1.0).variance(), 0.25 / 7.0, 1e-14);
        assert::close(new!(0.0, 0.3, 1.0).variance(), 0.033174603174603176, 1e-14);
        assert::close(new!(0.0, 0.9, 1.0).variance(), 0.02555555555555556, 1e-14);
    }
}
