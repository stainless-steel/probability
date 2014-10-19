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
    fn cdf(&self, _: f64) -> f64 {
        0.0
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
