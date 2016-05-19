use source::Source;
use distribution;

/// An exponential distribution.
#[derive(Clone, Copy)]
pub struct Exponential {
    lambda: f64,
}

impl Exponential {
    /// Create an exponential distribution with rate `lambda`.
    ///
    /// It should hold that `lambda > 0`.
    #[inline]
    pub fn new(lambda: f64) -> Exponential {
        should!(lambda > 0.0);
        Exponential { lambda: lambda }
    }

    /// Return the rate parameter.
    #[inline(always)]
    pub fn lambda(&self) -> f64 { self.lambda }
}

impl distribution::Distribution for Exponential {
    type Value = f64;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            -(-self.lambda * x).exp_m1()
        }
    }
}

impl distribution::Continuous for Exponential {
    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }
}

impl distribution::Entropy for Exponential {
    #[inline]
    fn entropy(&self) -> f64 {
        1.0 - self.lambda.ln()
    }
}

impl distribution::Inverse for Exponential {
    #[inline]
    fn inv_cdf(&self, p: f64) -> f64 {
        should!(0.0 <= p && p <= 1.0);
        -(-p).ln_1p() / self.lambda
    }
}

impl distribution::Kurtosis for Exponential {
    #[inline]
    fn kurtosis(&self) -> f64 { 6.0 }
}

impl distribution::Mean for Exponential {
    #[inline]
    fn mean(&self) -> f64 {
        self.lambda.recip()
    }
}

impl distribution::Median for Exponential {
    #[inline]
    fn median(&self) -> f64 {
        use std::f64::consts::LN_2;
        self.lambda.recip() * LN_2
    }
}

impl distribution::Modes for Exponential {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![0.0]
    }
}

impl distribution::Sample for Exponential {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64 where S: Source {
        -source.read::<f64>().ln() / self.lambda
    }
}

impl distribution::Skewness for Exponential {
    #[inline]
    fn skewness(&self) -> f64 { 2.0 }
}

impl distribution::Variance for Exponential {
    #[inline]
    fn variance(&self) -> f64 {
        self.lambda.powi(-2)
    }

    #[inline]
    fn deviation(&self) -> f64 {
        self.lambda.recip()
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($lambda:expr) => (Exponential::new($lambda));
    );

    #[test]
    fn cdf() {
        let d = new!(2.0);
        let x = vec![-1.0, 0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
        let p = vec![
            0.000000000000000e+00, 0.000000000000000e+00, 1.980132669324470e-02,
            9.516258196404043e-02, 1.812692469220182e-01, 2.591817793182821e-01,
            3.934693402873666e-01, 6.321205588285577e-01, 8.646647167633873e-01,
            9.502129316321360e-01, 9.816843611112658e-01, 9.975212478233336e-01,
            9.996645373720975e-01
        ];

        assert::close(&x.iter().map(|&x| d.cdf(x)).collect::<Vec<_>>(), &p, 1e-15);
    }

    #[test]
    fn pdf() {
        let d = new!(2.0);
        let x = vec![-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 12.0];
        let p = vec![
            0.000000000000000e+00, 2.000000000000000e+00, 7.357588823428847e-01,
            2.706705664732254e-01, 9.957413673572789e-02, 3.663127777746836e-02,
            1.347589399817093e-02, 4.957504353332717e-03, 6.709252558050237e-04,
            1.228842470665642e-05, 7.550269088558195e-11,
        ];

        assert::close(&x.iter().map(|&x| d.pdf(x)).collect::<Vec<_>>(), &p, 1e-15);
    }

    #[test]
    fn entropy() {
        use std::f64::consts::E;
        assert_eq!(new!(E).entropy(), 0.0);
    }

    #[test]
    fn inv_cdf() {
        use std::f64::INFINITY;

        let d = new!(2.0);
        let x = vec![
            0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, INFINITY,
        ];
        let p = vec![
            0.000000000000000e+00, 1.980132669324470e-02, 9.516258196404043e-02,
            1.812692469220182e-01, 2.591817793182821e-01, 3.934693402873666e-01,
            6.321205588285577e-01, 8.646647167633873e-01, 9.502129316321360e-01,
            9.816843611112658e-01, 9.975212478233336e-01, 9.996645373720975e-01,
            1.000000000000000e-00,
        ];

        assert::close(&p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>(), &x, 1e-14);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(2.0).kurtosis(), 6.0);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(2.0).mean(), 0.5);
    }

    #[test]
    fn median() {
        use std::f64::consts::LN_2;
        assert_eq!(new!(LN_2).median(), 1.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(2.0).modes(), vec![0.0]);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(2.0).skewness(), 2.0);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(2.0).variance(), 0.25);
    }

    #[test]
    fn deviation() {
        assert_eq!(new!(2.0).deviation(), 0.5);
    }
}
