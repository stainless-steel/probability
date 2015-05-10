use rand::distributions::{Normal, IndependentSample};

use {Distribution, Generator};

/// A Gaussian distribution.
#[derive(Clone, Copy)]
pub struct Gaussian {
    /// The mean value.
    pub mu: f64,
    /// The standard deviation.
    pub sigma: f64,

    normal: Normal,
}

impl Gaussian {
    /// Create a Gaussian distribution with the mean value `mu` and standard
    /// deviation `sigma`.
    #[inline]
    pub fn new(mu: f64, sigma: f64) -> Gaussian {
        Gaussian {
            mu: mu,
            sigma: sigma,
            normal: Normal::new(mu, sigma),
        }
    }
}

impl Distribution for Gaussian {
    type Item = f64;

    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        use std::f64::consts::PI;
        let var = self.sigma.powi(2);
        (-(x - self.mu).powi(2) / (2.0*var)).exp() / ((2.0*PI).sqrt() * self.sigma)
    }

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        use special::erf;
        use std::f64::consts::SQRT_2;
        (1.0 + erf((x - self.mu) / (self.sigma * SQRT_2))) / 2.0
    }

    /// Compute the inverse of the cumulative distribution function at
    /// probability `p`.
    ///
    /// The code is based on a [C implementation][1] by John Burkardt.
    ///
    /// [1]: http://people.sc.fsu.edu/~jburkardt/c_src/asa241/asa241.html
    fn inv_cdf(&self, p: f64) -> f64 {
        use ::num::Float;

        #[inline(always)]
        fn ln(x: f64) -> f64 { x.ln() }
        #[inline(always)]
        fn sqrt(x: f64) -> f64 { x.sqrt() }

        const CONST1: f64 = 0.180625;
        const CONST2: f64 = 1.6;
        const SPLIT1: f64 = 0.425;
        const SPLIT2: f64 = 5.0;
        const A: [f64; 8] = [
            3.3871328727963666080e+00, 1.3314166789178437745e+02,
            1.9715909503065514427e+03, 1.3731693765509461125e+04,
            4.5921953931549871457e+04, 6.7265770927008700853e+04,
            3.3430575583588128105e+04, 2.5090809287301226727e+03,
        ];
        const B: [f64; 8] = [
            1.0000000000000000000e+00, 4.2313330701600911252e+01,
            6.8718700749205790830e+02, 5.3941960214247511077e+03,
            2.1213794301586595867e+04, 3.9307895800092710610e+04,
            2.8729085735721942674e+04, 5.2264952788528545610e+03,
        ];
        const C: [f64; 8] = [
            1.42343711074968357734e+00, 4.63033784615654529590e+00,
            5.76949722146069140550e+00, 3.64784832476320460504e+00,
            1.27045825245236838258e+00, 2.41780725177450611770e-01,
            2.27238449892691845833e-02, 7.74545014278341407640e-04,
        ];
        const D: [f64; 8] = [
            1.00000000000000000000e+00, 2.05319162663775882187e+00,
            1.67638483018380384940e+00, 6.89767334985100004550e-01,
            1.48103976427480074590e-01, 1.51986665636164571966e-02,
            5.47593808499534494600e-04, 1.05075007164441684324e-09,
        ];
        const E: [f64; 8] = [
            6.65790464350110377720e+00, 5.46378491116411436990e+00,
            1.78482653991729133580e+00, 2.96560571828504891230e-01,
            2.65321895265761230930e-02, 1.24266094738807843860e-03,
            2.71155556874348757815e-05, 2.01033439929228813265e-07,
        ];
        const F: [f64; 8] = [
            1.00000000000000000000e+00, 5.99832206555887937690e-01,
            1.36929880922735805310e-01, 1.48753612908506148525e-02,
            7.86869131145613259100e-04, 1.84631831751005468180e-05,
            1.42151175831644588870e-07, 2.04426310338993978564e-15,
        ];

        #[inline(always)]
        fn poly(c: &[f64], x: f64) -> f64 {
            c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (
            c[4] + x * (c[5] + x * (c[6] + x * (c[7])))))))
        }

        if p <= 0.0 {
            return Float::neg_infinity();
        }
        if 1.0 <= p {
            return Float::infinity();
        }

        let q = p - 0.5;

        if (if q < 0.0 { -q } else { q }) <= SPLIT1 {
            let x = CONST1 - q * q;
            return self.mu + self.sigma * q * poly(&A, x) / poly(&B, x);
        }

        let mut x = if q < 0.0 { p } else { 1.0 - p };

        x = sqrt(-ln(x));

        if x <= SPLIT2 {
            x -= CONST2;
            x = poly(&C, x) / poly(&D, x);
        } else {
            x -= SPLIT2;
            x = poly(&E, x) / poly(&F, x);
        }

        self.mu + self.sigma * if q < 0.0 { -x } else { x }
    }

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> f64 {
        self.normal.ind_sample(generator)
    }
}

#[cfg(test)]
mod tests {
    use assert;

    use Distribution;
    use distributions::Gaussian;

    #[test]
    fn pdf() {
        let gaussian = Gaussian::new(1.0, 2.0);

        let x = vec![
            -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0
        ];
        let p = vec![
            8.764150246784270e-03, 1.586982591783371e-02, 2.699548325659403e-02,
            4.313865941325577e-02, 6.475879783294587e-02, 9.132454269451096e-02,
            1.209853622595717e-01, 1.505687160774022e-01, 1.760326633821498e-01,
            1.933340584014246e-01, 1.994711402007164e-01, 1.933340584014246e-01,
            1.760326633821498e-01, 1.505687160774022e-01, 1.209853622595717e-01,
            9.132454269451096e-02, 6.475879783294587e-02
        ];

        assert::within(&x.iter().map(|&x| gaussian.pdf(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn cdf() {
        let gaussian = Gaussian::new(1.0, 2.0);

        let x = vec![
            -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,
        ];
        let p = vec![
            6.209665325776139e-03, 1.222447265504470e-02, 2.275013194817922e-02,
            4.005915686381709e-02, 6.680720126885809e-02, 1.056497736668553e-01,
            1.586552539314571e-01, 2.266273523768682e-01, 3.085375387259869e-01,
            4.012936743170763e-01, 5.000000000000000e-01, 5.987063256829237e-01,
            6.914624612740131e-01, 7.733726476231317e-01, 8.413447460685429e-01,
            8.943502263331446e-01, 9.331927987311419e-01,
        ];

        assert::within(&x.iter().map(|&x| gaussian.cdf(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn inv_cdf() {
        use ::num::Float;

        let gaussian = Gaussian::new(-1.0, 0.25);

        let p = vec![
            0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
        ];
        let x = vec![
             Float::neg_infinity(), -1.411213406737868e+00, -1.320387891386150e+00,
            -1.259108347373447e+00, -1.210405308393228e+00, -1.168622437549020e+00,
            -1.131100128177010e+00, -1.096330116601892e+00, -1.063336775783950e+00,
            -1.031415336713768e+00, -1.000000000000000e+00, -9.685846632862315e-01,
            -9.366632242160501e-01, -9.036698833981082e-01, -8.688998718229899e-01,
            -8.313775624509796e-01, -7.895946916067714e-01, -7.408916526265525e-01,
            -6.796121086138498e-01, -5.887865932621319e-01,  Float::infinity(),
        ];

        assert::within(&p.iter().map(|&p| gaussian.inv_cdf(p)).collect::<Vec<_>>(), &x, 1e-14);
    }
}

#[cfg(test)]
mod benches {
    use test;

    use {Distribution, Sampler};
    use distributions::{Gaussian, Uniform};

    #[bench]
    fn cdf(bench: &mut test::Bencher) {
        let mut generator = ::generator();
        let gaussian = Gaussian::new(0.0, 1.0);
        let x = Sampler(&gaussian, &mut generator).take(1000).collect::<Vec<_>>();

        bench.iter(|| {
            test::black_box(x.iter().map(|&x| gaussian.cdf(x)).collect::<Vec<_>>())
        })
    }

    #[bench]
    fn inv_cdf(bench: &mut test::Bencher) {
        let mut generator = ::generator();
        let gaussian = Gaussian::new(0.0, 1.0);
        let uniform = Uniform::new(0.0, 1.0);
        let p = Sampler(&uniform, &mut generator).take(1000).collect::<Vec<_>>();

        bench.iter(|| {
            test::black_box(p.iter().map(|&p| gaussian.inv_cdf(p)).collect::<Vec<_>>())
        })
    }
}
