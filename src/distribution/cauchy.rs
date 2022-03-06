use distribution;
use source::Source;

/// A Cauchy distribution.
///
/// The Cauchy distribution (also known as Lorentz or Cauchy–Lorentz
/// distribution) is a continuous probability distribution with a location
/// parameter `x_0`, a scale parameter `gamma > 0`, and the following
/// probability density function:
///
/// `p(x) = 1 / (pi * gamma * (1 + ((x - x_0) / gamma)^2))`.
///
/// The distribution is long tailed and has no mean or variance. It is unimodal
/// with the mode at `x_0`, around which it is symmetric.
#[derive(Clone, Copy, Debug)]
pub struct Cauchy {
    x_0: f64,
    gamma: f64,
}

impl Cauchy {
    /// Create a Cauchy distribution with location `x_0` and scale `gamma`.
    ///
    /// It should hold that `gamma > 0`.
    #[inline]
    pub fn new(x_0: f64, gamma: f64) -> Self {
        should!(gamma > 0.0);
        Cauchy { x_0, gamma }
    }

    /// Return the location parameter.
    #[inline(always)]
    pub fn x_0(&self) -> f64 {
        self.x_0
    }

    /// Return the scale parameter.
    #[inline(always)]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl distribution::Continuous for Cauchy {
    #[inline]
    fn density(&self, x: f64) -> f64 {
        use std::f64::consts::PI;
        let deviation = x - self.x_0;
        self.gamma / (PI * (self.gamma * self.gamma + deviation * deviation))
    }
}

impl distribution::Distribution for Cauchy {
    type Value = f64;

    #[inline]
    fn distribution(&self, x: f64) -> f64 {
        use std::f64::consts::FRAC_1_PI;
        FRAC_1_PI * ((x - self.x_0) / self.gamma).atan() + 0.5
    }
}

impl distribution::Entropy for Cauchy {
    #[inline]
    fn entropy(&self) -> f64 {
        (std::f64::consts::PI * 4.0 * self.gamma).ln()
    }
}

impl distribution::Inverse for Cauchy {
    #[inline]
    fn inverse(&self, p: f64) -> f64 {
        use std::f64::{consts::PI, INFINITY, NEG_INFINITY};

        should!((0.0..=1.0).contains(&p));

        if p <= 0.0 {
            NEG_INFINITY
        } else if 1.0 <= p {
            INFINITY
        } else {
            self.x_0 + self.gamma * (PI * (p - 0.5)).tan()
        }
    }
}

impl distribution::Median for Cauchy {
    #[inline]
    fn median(&self) -> f64 {
        self.x_0
    }
}

impl distribution::Modes for Cauchy {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![self.x_0]
    }
}

impl distribution::Sample for Cauchy {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        let gaussian = distribution::Gaussian::new(0.0, 1.0);
        let a = gaussian.sample(source);
        let b = gaussian.sample(source);
        self.x_0() + self.gamma() * a / (b.abs() + f64::MIN_POSITIVE)
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($x_0:expr, $gamma:expr) => (Cauchy::new($x_0, $gamma));
    );

    #[test]
    fn density() {
        let d = new!(2.0, 8.0);
        let x = vec![-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 12.0];
        let p = vec![
            0.03488327519822364,
            0.03744822190397538,
            0.03843742021842001,
            0.039176601376466544,
            0.03963391578942141,
            0.039788735772973836,
            0.03963391578942141,
            0.039176601376466544,
            0.03744822190397538,
            0.03183098861837907,
            0.015527311521160521,
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
            0.3857997487800918,
            0.4220208696226307,
            0.4223954618429798,
            0.4238960166273086,
            0.4257765641957529,
            0.42766240385764065,
            0.43144951512041,
            0.44100191513247144,
            0.46041657583943446,
            0.48013147569445913,
            0.5,
            0.5395834241605656,
            0.5779791303773694,
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
        assert_eq!(new!(2.0, 1.0).entropy(), (PI * 4.0).ln());
        assert::close(new!(3.0, 5.2).entropy(), 4.1796828725566719243, 1e-15);
    }

    #[test]
    fn inverse() {
        use std::f64::{INFINITY, NEG_INFINITY};

        let d = new!(2.0, 3.0);
        let x = vec![
            NEG_INFINITY,
            -7.2330506115257585,
            -0.9999999999999996,
            2.0,
            5.0,
            11.233050611525758,
            INFINITY,
        ];
        let p = vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

        assert::close(
            &p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(),
            &x,
            1e-14,
        );

        assert!(d.inverse(0.0) < -1e16);
        assert!(d.inverse(1.0) > 1e16);
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
    fn sampling() {
        let n = 100000;
        let d = Cauchy::new(35.4, 12.3);
        let mut source = source::Xorshift128Plus::new([42, 69]);

        let cross_entropy = -(0..n)
            .map(|_| d.density(d.sample(&mut source)).ln())
            .sum::<f64>()
            / n as f64;
        assert!((cross_entropy - d.entropy()).abs() < 0.01);
    }
}
