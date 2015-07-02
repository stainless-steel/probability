use random::Source;

/// A gamma distribution.
#[derive(Clone, Copy)]
pub struct Gamma {
    alpha: f64,
    beta: f64,
}

impl Gamma {
    /// Create a gamma distribution with shape parameter `alpha` and rate
    /// parameter `beta`.
    ///
    /// It should hold that `alpha > 0` and `beta > 0`.
    #[inline]
    pub fn new(alpha: f64, beta: f64) -> Gamma {
        should!(alpha > 0.0 && beta > 0.0);
        Gamma { alpha: alpha, beta: beta }
    }

    /// Return the shape parameter.
    #[inline(always)] pub fn alpha(&self) -> f64 { self.alpha }

    /// Return the rate parameter.
    #[inline(always)] pub fn beta(&self) -> f64 { self.beta }

    /// Draw a sample.
    ///
    /// ## References
    ///
    /// 1. G. Marsaglia and W. W. Tsang, “A simple method for generating gamma
    ///    variables,” ACM Transactions on Mathematical Software, vol. 26,
    ///    no. 3, pp. 363–372, September 2000.
    #[inline]
    pub fn sample<S>(&self, source: &mut S) -> f64 where S: Source {
        sample(self.alpha, self.beta, source)
    }
}

/// Draw a sample from a Gamma distribution.
pub fn sample<S: Source>(alpha: f64, beta: f64, source: &mut S) -> f64 {
    use distribution::gaussian;

    if alpha < 1.0 {
        return sample(1.0 + alpha, beta, source) * source.take::<f64>().powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
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
            let u = source.take::<f64>();
            if u == 0.0 {
                continue;
            }

            if u < 1.0 - 0.0331 * x * x {
                return beta * d * v;
            }
            if u.ln() < 0.5 * x + d * (1.0 - v + v.ln()) {
                return beta * d * v;
            }

            break;
        }
    }
}
