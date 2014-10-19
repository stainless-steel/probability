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
    fn cdf(&self, x: f64) -> f64 {
        use self::sfunc::erf;
        (1.0 + erf((x - self.mu) / (self.sigma * Float::sqrt2()))) / 2.0
    }

    fn inv_cdf(&self, p: f64) -> f64 {
        use self::sfunc::{log, sqrt};

        const CONST1: f64 = 0.180625;
        const CONST2: f64 = 1.6;
        const SPLIT1: f64 = 0.425;
        const SPLIT2: f64 = 5.0;

        const A: [f64, ..8] = [
            3.3871328727963666080,
            1.3314166789178437745e+2,
            1.9715909503065514427e+3,
            1.3731693765509461125e+4,
            4.5921953931549871457e+4,
            6.7265770927008700853e+4,
            3.3430575583588128105e+4,
            2.5090809287301226727e+3,
        ];
        const B: [f64, ..8] = [
            1.0,
            4.2313330701600911252e+1,
            6.8718700749205790830e+2,
            5.3941960214247511077e+3,
            2.1213794301586595867e+4,
            3.9307895800092710610e+4,
            2.8729085735721942674e+4,
            5.2264952788528545610e+3,
        ];
        const C: [f64, ..8] = [
            1.42343711074968357734,
            4.63033784615654529590,
            5.76949722146069140550,
            3.64784832476320460504,
            1.27045825245236838258,
            2.41780725177450611770e-1,
            2.27238449892691845833e-2,
            7.74545014278341407640e-4,
        ];
        const D: [f64, ..8] = [
            1.0,
            2.05319162663775882187,
            1.67638483018380384940,
            6.89767334985100004550e-1,
            1.48103976427480074590e-1,
            1.51986665636164571966e-2,
            5.47593808499534494600e-4,
            1.05075007164441684324e-9,
        ];
        const E: [f64, ..8] = [
            6.65790464350110377720,
            5.46378491116411436990,
            1.78482653991729133580,
            2.96560571828504891230e-1,
            2.65321895265761230930e-2,
            1.24266094738807843860e-3,
            2.71155556874348757815e-5,
            2.01033439929228813265e-7,
        ];
        const F: [f64, ..8] = [
            1.0,
            5.99832206555887937690e-1,
            1.36929880922735805310e-1,
            1.48753612908506148525e-2,
            7.86869131145613259100e-4,
            1.84631831751005468180e-5,
            1.42151175831644588870e-7,
            2.04426310338993978564e-15,
        ];

        #[inline]
        fn poly(c: &[f64], x: f64) -> f64 {
            c.iter().rev().fold(0.0, |y, &c| y * x + c)
        }

        if p <= 0.0 {
            return Float::neg_infinity();
        }
        if 1.0 <= p {
            return Float::infinity();
        }

        let q = p - 0.5;

        if ::std::num::abs(q) <= SPLIT1 {
            let x = CONST1 - q * q;
            return self.mu + self.sigma * q * poly(&A, x) / poly(&B, x);
        }

        let mut x = if q < 0.0 { p } else { 1.0 - p };

        x = sqrt(-log(x));

        if x <= SPLIT2 {
            x -= CONST2;
            x = poly(&C, x) / poly(&D, x);
        } else {
            x -= SPLIT2;
            x = poly(&E, x) / poly(&F, x);
        }

        if q < 0.0 {
            x = -x;
        }

        self.mu + self.sigma * x
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
                if x[i].is_finite() {
                    assert!(::std::num::abs(x[i] - y[i]) < e,
                            "expected {:e} ~ {:e}", x[i], y[i]);
                } else {
                    assert_eq!(x[i], y[i]);
                }
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

    #[test]
    fn inv_cdf() {
        let gaussian = Gaussian::new(-1.0, 0.25);

        let p = vec![0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                     0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                     0.90, 0.95, 1.00];
        let x = vec![Float::neg_infinity(), -1.411213406737868e+00,
                     -1.320387891386150e+00, -1.259108347373447e+00,
                     -1.210405308393228e+00, -1.168622437549020e+00,
                     -1.131100128177010e+00, -1.096330116601892e+00,
                     -1.063336775783950e+00, -1.031415336713768e+00,
                     -1.000000000000000e+00, -9.685846632862315e-01,
                     -9.366632242160501e-01, -9.036698833981082e-01,
                     -8.688998718229899e-01, -8.313775624509796e-01,
                     -7.895946916067714e-01, -7.408916526265525e-01,
                     -6.796121086138498e-01, -5.887865932621319e-01,
                     Float::infinity()];

        assert_almost_eq!(p.iter().map(|&p| {
            gaussian.inv_cdf(p)
        }).collect(), x);
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
