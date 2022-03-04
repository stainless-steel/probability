use distribution;
use source::Source;

/// A Cauchy distribution.
///
/// A Cauchy distribution (aka Lorentz or Cauchy–Lorentz distribution) is a continuous
/// probability distribution with a location parameter `loc`, a scale parameter `gamma > 0`,
/// and the following probability density function:
///
/// `p(x) = const / (1 + ((x - loc) / gamma)^2)`.
///
/// A Cauchy distribution is long tailed and has no well-defined mean or variance. It is
/// unimodal with its mode at `loc`, around which it is symmetric. The ratio of two
/// independent Gaussian distributed random variables is Cauchy distributed.
///
/// See [Wikipedia article on Cauchy
/// distribution](https://en.wikipedia.org/wiki/Cauchy_distribution).
#[derive(Clone, Copy, Debug)]
pub struct Cauchy {
    loc: f64,
    gamma: f64,
}

impl Cauchy {
    /// Create a Cauchy distribution with location `loc` and scale `gamma`.
    ///
    /// It should hold that `gamma > 0`.
    #[inline]
    pub fn new(loc: f64, gamma: f64) -> Self {
        should!(gamma > 0.0);
        Cauchy { loc, gamma }
    }

    // Return the location parameter
    #[inline(always)]
    pub fn loc(&self) -> f64 {
        self.loc
    }

    // Return the scale parameter
    #[inline(always)]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl distribution::Continuous for Cauchy {
    #[inline]
    fn density(&self, x: f64) -> f64 {
        // Divisions are expensive, so we implement the PDF with only a single division.
        use std::f64::consts::PI;
        let deviation = x - self.loc;
        self.gamma / (PI * (self.gamma * self.gamma + deviation * deviation))
    }
}

impl distribution::Distribution for Cauchy {
    type Value = f64;

    #[inline]
    fn distribution(&self, x: f64) -> f64 {
        use std::f64::consts::FRAC_1_PI;
        FRAC_1_PI * ((x - self.loc) / self.gamma).atan() + 0.5
    }
}

impl distribution::Entropy for Cauchy {
    #[inline]
    fn entropy(&self) -> f64 {
        (std::f64::consts::PI * 4.0 * self.gamma).ln()
    }
}

impl distribution::Inverse for Cauchy {
    /// Due to finite precision of arithmetic operations, the current implementation of
    /// `Inverse::inverse` for `Cauchy` does *not* return negative or positive infinity for
    /// `p = 0.0` or `p = 1.0`, respectively. It instead returns very large (in magnitude)
    /// values (`> 1e16`).
    #[inline]
    fn inverse(&self, p: f64) -> f64 {
        should!((0.0..=1.0).contains(&p));
        use std::f64::consts::PI;
        self.loc + self.gamma * (PI * (p - 0.5)).tan()
    }
}

impl distribution::Median for Cauchy {
    #[inline]
    fn median(&self) -> f64 {
        self.loc
    }
}

impl distribution::Modes for Cauchy {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![self.loc]
    }
}

impl distribution::Sample for Cauchy {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        // We use the fact that the ratio of two standard normal random variables is
        // standard Cauchy distributed. This way of drawing a sample from a Cauchy
        // distribution turned out to be more efficient in the benchmarks than the naive way
        // of applying the quantile function to a uniformly distributed random variable.
        // However, the employed method requires reading more random numbers from the
        // source, so it might be slower for slow random sources.
        let gaussian = distribution::Gaussian::new(0.0, 1.0);
        let a = gaussian.sample(source);
        let b = gaussian.sample(source);
        self.loc() + self.gamma() * a / (b.abs() + f64::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($loc:expr, $gamma:expr) => (Cauchy::new($loc, $gamma));
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
        let d = new!(2.0, 3.0);
        let x = vec![
            -7.2330506115257585,
            -0.9999999999999996,
            2.0,
            5.0,
            11.233050611525758,
        ];
        let p = vec![0.1, 0.25, 0.5, 0.75, 0.9];

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

        // Estimate the Kullback-Leibler divergence `KL(samples || d)` based on
        // `n` samples and assert that the estimate is close to zero.
        // This test is reproducible because we use a fixed random seed.
        let cross_entropy_estimate = -(0..n)
            .map(|_| d.density(d.sample(&mut source)).ln())
            .sum::<f64>()
            / n as f64;
        let kl_divergence_estimate = cross_entropy_estimate - d.entropy();
        assert!(kl_divergence_estimate.abs() < 0.01);
    }
}
