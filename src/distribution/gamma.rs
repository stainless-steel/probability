use generator::Generator;

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
    /// 1. George Marsaglia and Wai Wan Tsang, “A Simple Method for Generating
    ///    Gamma Variables,” ACM Transactions on Mathematical Software, vol. 26,
    ///    no. 3, 2000, pp. 363–372.
    #[inline(always)]
    pub fn sample<G>(&self, generator: &mut G) -> f64 where G: Generator {
        sample(self.alpha, self.beta, generator)
    }
}

/// Draw a sample from a Gamma distribution.
pub fn sample<G: Generator>(alpha: f64, beta: f64, generator: &mut G) -> f64 {
    use distribution::gaussian;

    if alpha < 1.0 {
        return sample(1.0 + alpha, beta, generator) * generator.next::<f64>().powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();

    loop {
        let mut x = gaussian::sample(generator);
        let mut v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }

        x = x * x;
        v = v * v * v;

        loop {
            let u = generator.next::<f64>();
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
