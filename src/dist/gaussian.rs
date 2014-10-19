extern crate sfunc;

use std::rand::Rng;
use std::rand::distributions::{Normal, IndependentSample};

use super::Distribution;

/// A Gaussian distribution with a mean value `mu` and a standard deviation
/// `sigma`.
pub struct Gaussian {
    /// The mean value.
    pub mu: f64,
    /// The standard deviation.
    pub sigma: f64,

    normal: Normal,
}

impl Gaussian {
    /// Creates a Gaussian distribution with the mean value `mu` and the
    /// standard deviation `sigma`.
    #[inline]
    pub fn new(mu: f64, sigma: f64) -> Gaussian {
        Gaussian {
            mu: mu,
            sigma: sigma,
            normal: Normal::new(0.0, 1.0),
        }
    }
}

impl Distribution<f64> for Gaussian {
    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        use self::sfunc::erf;
        (1.0 + erf((x - self.mu) / (self.sigma * Float::sqrt2()))) / 2.0
    }

    #[inline]
    fn inv_cdf(&self, _: f64) -> f64 {
        0.0
    }

    #[inline]
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        self.normal.ind_sample(rng)
    }
}

#[cfg(test)]
mod test {
    use super::super::Distribution;
    use super::Gaussian;

    macro_rules! assert_almost_eq(
        ($x:expr, $y:expr) => ({
            let e: f64 = ::std::f64::EPSILON.sqrt();
            let x: Vec<f64> = $x;
            let y: Vec<f64> = $y;
            for i in range(0u, x.len()) {
                assert!(::std::num::abs(x[i] - y[i]) < e,
                        "expected {:e} ~ {:e}", x[i], y[i]);
            }
        });
    )

    #[test]
    fn cdf() {
        let gaussian = Gaussian::new(1.0, 2.0);

        let x = vec![-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5,
                     1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let p = vec![6.209665325776139e-03, 1.222447265504470e-02,
                     2.275013194817922e-02, 4.005915686381709e-02,
                     6.680720126885809e-02, 1.056497736668553e-01,
                     1.586552539314571e-01, 2.266273523768682e-01,
                     3.085375387259869e-01, 4.012936743170763e-01,
                     5.000000000000000e-01, 5.987063256829237e-01,
                     6.914624612740131e-01, 7.733726476231317e-01,
                     8.413447460685429e-01, 8.943502263331446e-01,
                     9.331927987311419e-01];

        assert_almost_eq!(x.iter().map(|&x| {
            gaussian.cdf(x)
        }).collect(), p);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    use super::super::Distribution;
    use super::Gaussian;

    #[bench]
    fn sample(bench: &mut test::Bencher) {
        let mut rng = ::std::rand::task_rng();
        let gaussian = Gaussian::new(0.0, 1.0);
        bench.iter(|| {
            test::black_box(gaussian.sample(&mut rng))
        });
    }
}
