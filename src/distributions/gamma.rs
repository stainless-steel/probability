use Generator;

/// A gamma distribution.
#[derive(Clone, Copy)]
pub struct Gamma {
    /// The shape parameter.
    alpha: f64,
    /// The rate parameter.
    beta: f64,
}

impl Gamma {
    /// Create a gamma distribution with the shape and rate parameters `alpha`
    /// and `beta`, respectively.
    ///
    /// ## Panics
    ///
    /// Panics if `alpha <= 0` or `beta <= 0`.
    #[inline]
    pub fn new(alpha: f64, beta: f64) -> Gamma {
        should!(alpha > 0.0 && beta > 0.0);
        Gamma { alpha: alpha, beta: beta }
    }

    #[inline(always)]
    pub fn sample<G: Generator>(&self, generator: &mut G) -> f64 {
        sample(self.alpha, self.beta, generator)
    }
}

// References:
//
// [1] G. Marsaglia and W. Tsang, “A Simple Method for generating gamma
//     variables,” ACM Transactions on Mathematical Software, vol. 26, no. 3,
//     2000, pp. 363–372.
fn sample<G: Generator>(alpha: f64, beta: f64, generator: &mut G) -> f64 {
    if alpha < 1.0 {
        return sample(1.0 + alpha, beta, generator) * generator.uniform().powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();

    loop {
        let mut x = generator.gaussian();
        let mut v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }

        x = x * x;
        v = v * v * v;

        loop {
            let u = generator.uniform();
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
