use distribution;
use source::Source;

/// A gamma distribution.
#[derive(Clone, Copy)]
pub struct Gamma {
    k: f64,
    theta: f64,
    norm: f64,
}

impl Gamma {
    /// Create a gamma distribution with shape parameter `k` and scale parameter
    /// `theta`.
    ///
    /// It should hold that `k > 0` and `theta > 0`.
    #[inline]
    pub fn new(k: f64, theta: f64) -> Gamma {
        use special::Gamma as SpecialGamma;
        should!(k > 0.0 && theta > 0.0);
        Gamma { k: k, theta: theta, norm: k.gamma() * theta.powf(k) }
    }

    /// Return the shape parameter.
    #[inline(always)]
    pub fn k(&self) -> f64 { self.k }

    /// Return the scale parameter.
    #[inline(always)]
    pub fn theta(&self) -> f64 { self.theta }
}

impl distribution::Continuous for Gamma {
    fn density(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            x.powf(self.k - 1.0) * (-x / self.theta).exp() / self.norm
        }
    }
}

impl distribution::Distribution for Gamma {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        use special::Gamma;
        if x <= 0.0 {
            0.0
        } else {
            (x / self.theta).inc_gamma(self.k)
        }
    }
}

impl distribution::Mean for Gamma {
    #[inline]
    fn mean(&self) -> f64 {
        self.k * self.theta
    }
}

impl distribution::Variance for Gamma {
    #[inline]
    fn variance(&self) -> f64 {
        self.k * self.theta * self.theta
    }
}

impl distribution::Sample for Gamma {
    /// Draw a sample.
    ///
    /// ## References
    ///
    /// 1. G. Marsaglia and W. W. Tsang, “A simple method for generating gamma
    ///    variables,” ACM Transactions on Mathematical Software, vol. 26,
    ///    no. 3, pp. 363–372, September 2000.
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64 where S: Source {
        self.theta * sample(self.k, source)
    }
}

/// Draw a sample from the standard Gamma distribution.
pub fn sample<S: Source>(k: f64, source: &mut S) -> f64 {
    use distribution::gaussian;

    if k < 1.0 {
        return sample(1.0 + k, source) * source.read::<f64>().powf(1.0 / k);
    }

    let d = k - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();

    loop {
        let mut x = gaussian::sample(source);
        let mut v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }

        x = x * x;
        v = v * v * v;

        loop {
            let u = source.read::<f64>();
            if u == 0.0 {
                continue;
            }

            if u < 1.0 - 0.0331 * x * x {
                return d * v;
            }
            if u.ln() < 0.5 * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }

            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($k:expr, $theta:expr) => (Gamma::new($k, $theta));
    );

    #[test]
    fn density() {
        let d = new!(9.0, 0.5);
        let x = vec![
            -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
             4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
        ];
        let p = vec![
            0.0000000000000000e+00, 0.0000000000000000e+00, 1.8247988153345374e-05,
            1.7185432791950819e-03, 1.6203023589362864e-02, 5.9540362609726317e-02,
            1.3055607869631736e-01, 2.0651546706168886e-01, 2.6075486443009133e-01,
            2.7917306390119390e-01, 2.6351128001904534e-01, 2.2519806429803990e-01,
            1.7758719703342618e-01, 1.3104656985457416e-01, 9.1459332185226061e-02,
            6.0871080525929738e-02, 3.8888600663684249e-02, 2.3974945191907938e-02,
            1.4325000668200492e-02, 8.3250881130958170e-03,
        ];

        assert::close(&x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn distribution() {
        let d = new!(9.0, 0.5);
        let x = vec![
            -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0
        ];
        let p = vec![
            0.000000000000000e+00, 0.000000000000000e+00, 2.374473282611617e-04,
            2.136343448798417e-02, 1.527625060154386e-01, 4.074526585624087e-01,
            6.671803212492811e-01, 8.449722182325371e-01, 9.379448040996508e-01,
            9.780127464509413e-01, 9.929439908525065e-01, 9.979127409508650e-01,
            9.994230988333771e-01, 9.998494373023702e-01, 9.999625848537679e-01,
            9.999910875735207e-01, 9.999979539240957e-01, 9.999995452593557e-01,
            9.999999017950048e-01, 9.999999793279283e-01, 9.999999957473356e-01,
        ];

        assert::close(&x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(9.0, 0.5).mean(), 4.5);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(9.0, 0.5).variance(), 2.25);
    }
}
