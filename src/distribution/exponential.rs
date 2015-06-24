use distribution::Distribution;
use generator::Generator;

/// An exponential distribution.
#[derive(Clone, Copy)]
pub struct Exponential {
    /// The rate parameter.
    pub lambda: f64,
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
}

impl Distribution for Exponential {
    type Value = f64;

    #[inline] fn mean(&self) -> f64 { self.lambda.recip() }
    #[inline] fn var(&self) -> f64 { self.lambda.powi(-2) }
    #[inline] fn sd(&self) -> f64 { self.lambda.recip() }
    #[inline] fn skewness(&self) -> f64 { 2.0 }
    #[inline] fn kurtosis(&self) -> f64 { 6.0 }

    #[inline]
    fn median(&self) -> f64 {
        use std::f64::consts::LN_2;
        self.lambda.recip() * LN_2
    }

    #[inline] fn modes(&self) -> Vec<f64> { vec![0.0] }
    #[inline] fn entropy(&self) -> f64 { 1.0 - self.lambda.ln() }

    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.lambda * (-self.lambda * x).exp() }
    }

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { -(-self.lambda * x).exp_m1() }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> f64 {
        should!(0.0 <= p && p <= 1.0);
        -(-p).ln_1p() / self.lambda
    }

    #[inline(always)]
    fn sample<G: Generator>(&self, generator: &mut G) -> f64 {
        -generator.next::<f64>().ln() / self.lambda
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
    fn mean() {
        assert_eq!(new!(2.0).mean(), 0.5);
    }

    #[test]
    fn var() {
        assert_eq!(new!(2.0).var(), 0.25);
    }

    #[test]
    fn sd() {
        assert_eq!(new!(2.0).sd(), 0.5);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(2.0).skewness(), 2.0);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(2.0).kurtosis(), 6.0);
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
    fn entropy() {
        use std::f64::consts::E;
        assert_eq!(new!(E).entropy(), 0.0);
    }

    #[test]
    fn pdf() {
        let exponential = new!(2.0);
        let x = vec![-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 12.0];
        let p = vec![
            0.000000000000000e+00, 2.000000000000000e+00, 7.357588823428847e-01,
            2.706705664732254e-01, 9.957413673572789e-02, 3.663127777746836e-02,
            1.347589399817093e-02, 4.957504353332717e-03, 6.709252558050237e-04,
            1.228842470665642e-05, 7.550269088558195e-11,
        ];

        assert::close(&x.iter().map(|&x| exponential.pdf(x)).collect::<Vec<_>>(), &p, 1e-15);
    }

    #[test]
    fn cdf() {
        let exponential = new!(2.0);
        let x = vec![-1.0, 0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
        let p = vec![
            0.000000000000000e+00, 0.000000000000000e+00, 1.980132669324470e-02,
            9.516258196404043e-02, 1.812692469220182e-01, 2.591817793182821e-01,
            3.934693402873666e-01, 6.321205588285577e-01, 8.646647167633873e-01,
            9.502129316321360e-01, 9.816843611112658e-01, 9.975212478233336e-01,
            9.996645373720975e-01
        ];

        assert::close(&x.iter().map(|&x| exponential.cdf(x)).collect::<Vec<_>>(), &p, 1e-15);
    }

    #[test]
    fn inv_cdf() {
        use std::f64::INFINITY;

        let exponential = new!(2.0);
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

        assert::close(&p.iter().map(|&p| exponential.inv_cdf(p)).collect::<Vec<_>>(), &x, 1e-14);
    }
}
