use distribution;
use source::Source;

/// A logistic distribution.
#[derive(Clone, Copy)]
pub struct Logistic {
    mu: f64,
    s: f64,
}

impl Logistic {
    /// Create a logistic distribution with location `mu` and scale `s`.
    ///
    /// It should hold that `s > 0`.
    #[inline]
    pub fn new(mu: f64, s: f64) -> Self {
        should!(s > 0.0);
        Logistic { mu: mu, s: s }
    }

    /// Return the location parameter.
    #[inline(always)]
    pub fn mu(&self) -> f64 { self.mu }

    /// Return the scale parameter.
    #[inline(always)]
    pub fn s(&self) -> f64 { self.s }
}

impl Default for Logistic {
    #[inline]
    fn default() -> Self {
        Logistic::new(0.0, 1.0)
    }
}

impl distribution::Continuous for Logistic {
    #[inline]
    fn density(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-(x - self.mu) / self.s).exp())
    }
}

impl distribution::Distribution for Logistic {
    type Value = f64;

    #[inline]
    fn distribution(&self, x: f64) -> f64 {
        let exp = (-(x - self.mu) / self.s).exp();
        exp / (self.s * (1.0 + exp).powi(2))
    }
}

impl distribution::Entropy for Logistic {
    #[inline]
    fn entropy(&self) -> f64 {
        self.s.ln() + 2.0
    }
}

impl distribution::Inverse for Logistic {
    #[inline]
    fn inverse(&self, p: f64) -> f64 {
        should!(0.0 <= p && p <= 1.0);
        self.mu - self.s * (1.0 / p - 1.0).ln()
    }
}

impl distribution::Kurtosis for Logistic {
    #[inline]
    fn kurtosis(&self) -> f64 { 1.2 }
}

impl distribution::Mean for Logistic {
    #[inline]
    fn mean(&self) -> f64 { self.mu }
}

impl distribution::Median for Logistic {
    #[inline]
    fn median(&self) -> f64 { self.mu }
}

impl distribution::Modes for Logistic {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![self.mu]
    }
}

impl distribution::Sample for Logistic {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64 where S: Source {
        use distribution::Inverse;
        self.inverse(source.read::<f64>())
    }
}

impl distribution::Skewness for Logistic {
    #[inline]
    fn skewness(&self) -> f64 { 0.0 }
}

impl distribution::Variance for Logistic {
    #[inline]
    fn variance(&self) -> f64 {
        use std::f64::consts::PI;
        (PI * self.s).powi(2) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;
    use std::f64::{INFINITY, NEG_INFINITY};

    macro_rules! new(
        ($mu:expr, $s:expr) => (Logistic::new($mu, $s));
    );

    #[test]
    fn density() {
        let d = new!(5.0, 5.0);
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = vec![
            2.6894142136999510e-01, 3.1002551887238755e-01, 3.5434369377420455e-01,
            4.0131233988754800e-01, 4.5016600268752216e-01, 5.0000000000000000e-01,
            5.4983399731247795e-01, 5.9868766011245200e-01, 6.4565630622579540e-01,
            6.8997448112761250e-01, 7.3105857863000490e-01,
        ];

        assert::close(&x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(), &p, 1e-15);
    }

    #[test]
    fn distribution() {
        let d = new!(5.0, 5.0);
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = vec![
            3.9322386648296369e-02, 4.2781939304058887e-02, 4.5756848091331452e-02,
            4.8052149148305828e-02, 4.9503314542371987e-02, 5.0000000000000003e-02,
            4.9503314542371987e-02, 4.8052149148305828e-02, 4.5756848091331452e-02,
            4.2781939304058887e-02, 3.9322386648296369e-02,
        ];

        assert::close(&x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(), &p, 1e-7);
    }

    #[test]
    fn entropy() {
        assert_eq!(new!(0.0, (-2f64).exp()).entropy(), 0.0);
    }

    #[test]
    fn inverse() {
        let d = new!(5.0, 5.0);
        let p = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let x = vec![
                      NEG_INFINITY, -5.9861228866810947e+00, -1.9314718055994531e+00,
            7.6351069806398275e-01,  2.9726744594591787e+00,  5.0000000000000000e+00,
            7.0273255405408239e+00,  9.2364893019360199e+00,  1.1931471805599454e+01,
            1.5986122886681098e+01,  INFINITY,
        ];

        assert::close(&p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(), &x, 1e-14);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(2.0, 1.0).kurtosis(), 1.2);
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
        use std::f64::consts::PI;
        assert_eq!(new!(1.0, 3.0 / PI).variance(), 3.0);
    }

    #[test]
    fn deviation() {
        use std::f64::consts::PI;
        assert_eq!(new!(1.0, 3.0 / PI).deviation(), 3f64.sqrt());
    }
}
