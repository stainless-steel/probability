use rand::distributions::{Gamma, IndependentSample};

use {Distribution, Generator};

/// A beta distribution.
pub struct Beta {
    /// The first shape parameter.
    pub alpha: f64,
    /// The second shape parameter.
    pub beta: f64,
    /// The left endpoint of the support.
    pub a: f64,
    /// The right endpoint of the support.
    pub b: f64,

    ln_beta: f64,
    gamma_alpha: Gamma,
    gamma_beta: Gamma,
}

impl Beta {
    /// Create a beta distribution with the shape parameters `alpha` and `beta`
    /// on the interval `[a, b]`.
    #[inline]
    pub fn new(alpha: f64, beta: f64, a: f64, b: f64) -> Beta {
        use special::ln_beta;
        Beta {
            alpha: alpha,
            beta: beta,
            a: a,
            b: b,
            ln_beta: ln_beta(alpha, beta),
            gamma_alpha: Gamma::new(alpha, 1.0),
            gamma_beta: Gamma::new(beta, 1.0),
        }
    }
}

impl Distribution for Beta {
    type Item = f64;

    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        let norm = self.b - self.a;
        let x = (x - self.a) / (self.b - self.a);
        ((self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() -
            self.ln_beta).exp() / norm
    }

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        use special::inc_beta;
        inc_beta((x - self.a) / (self.b - self.a), self.alpha, self.beta, self.ln_beta)
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> f64 {
        use special::inv_inc_beta;
        self.a + (self.b - self.a) * inv_inc_beta(p, self.alpha, self.beta, self.ln_beta)
    }

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> f64 {
        let x = self.gamma_alpha.ind_sample(generator);
        let y = self.gamma_beta.ind_sample(generator);
        self.a + (self.b - self.a) * x / (x + y)
    }
}

#[cfg(test)]
mod tests {
    use assert;

    use {Distribution, Sampler, generator};
    use distributions::Beta;

    #[test]
    fn pdf() {
        let beta = Beta::new(2.0, 3.0, -1.0, 2.0);

        let x = vec![
            -1.00, -0.85, -0.70, -0.55, -0.40, -0.25, -0.10, 0.05, 0.20, 0.35, 0.50,
             0.65,  0.80,  0.95,  1.10,  1.25,  1.40,  1.55, 1.70, 1.85, 2.00,
        ];
        let p = vec![
            0.000000000000000e+00, 1.805000000000000e-01, 3.240000000000001e-01,
            4.335000000000000e-01, 5.120000000000000e-01, 5.625000000000001e-01,
            5.880000000000000e-01, 5.915000000000001e-01, 5.760000000000001e-01,
            5.445000000000000e-01, 5.000000000000001e-01, 4.455000000000000e-01,
            3.840000000000001e-01, 3.184999999999999e-01, 2.519999999999999e-01,
            1.875000000000000e-01, 1.280000000000001e-01, 7.650000000000003e-02,
            3.600000000000000e-02, 9.499999999999982e-03, 0.000000000000000e+00
        ];

        assert::within(&x.iter().map(|&x| beta.pdf(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn cdf() {
        let beta = Beta::new(2.0, 3.0, -1.0, 2.0);

        let x = vec![
            -1.00, -0.85, -0.70, -0.55, -0.40, -0.25, -0.10, 0.05, 0.20, 0.35, 0.50,
             0.65,  0.80,  0.95,  1.10,  1.25,  1.40,  1.55, 1.70, 1.85, 2.00,
        ];
        let p = vec![
            0.000000000000000e+00, 1.401875000000000e-02, 5.230000000000002e-02,
            1.095187500000000e-01, 1.807999999999999e-01, 2.617187500000001e-01,
            3.483000000000000e-01, 4.370187500000001e-01, 5.248000000000003e-01,
            6.090187500000001e-01, 6.875000000000000e-01, 7.585187500000001e-01,
            8.208000000000000e-01, 8.735187499999999e-01, 9.163000000000000e-01,
            9.492187500000000e-01, 9.728000000000000e-01, 9.880187500000001e-01,
            9.963000000000000e-01, 9.995187500000000e-01, 1.000000000000000e+00,
        ];

        assert::within(&x.iter().map(|&x| beta.cdf(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn inv_cdf() {
        let beta = Beta::new(1.0, 2.0, 3.0, 4.0);

        let p = vec![
            0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
        ];
        let x = vec![
            3.000000000000000e+00, 3.025320565519104e+00, 3.051316701949486e+00,
            3.078045554270711e+00, 3.105572809000084e+00, 3.133974596215561e+00,
            3.163339973465924e+00, 3.193774225170145e+00, 3.225403330758517e+00,
            3.258380151290432e+00, 3.292893218813452e+00, 3.329179606750063e+00,
            3.367544467966324e+00, 3.408392021690038e+00, 3.452277442494834e+00,
            3.500000000000000e+00, 3.552786404500042e+00, 3.612701665379257e+00,
            3.683772233983162e+00, 3.776393202250021e+00, 4.000000000000000e+00,
        ];

        assert::within(&p.iter().map(|&p| beta.inv_cdf(p)).collect::<Vec<_>>(), &x, 1e-14);
    }

    #[test]
    fn sample() {
        for x in Sampler(&Beta::new(1.0, 2.0, 7.0, 42.0), &mut generator()).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }
}

#[cfg(test)]
mod benches {
    use test;

    use {Distribution, Sampler, generator};
    use distributions::{Beta, Uniform};

    #[bench]
    fn cdf(bench: &mut test::Bencher) {
        let beta = Beta::new(0.5, 1.5, 0.0, 1.0);
        let x = Sampler(&beta, &mut generator()).take(1000).collect::<Vec<_>>();

        bench.iter(|| {
            test::black_box(x.iter().map(|&x| beta.cdf(x)).collect::<Vec<_>>())
        });
    }

    #[bench]
    fn inv_cdf(bench: &mut test::Bencher) {
        let beta = Beta::new(0.5, 1.5, 0.0, 1.0);
        let uniform = Uniform::new(0.0, 1.0);
        let p = Sampler(&uniform, &mut generator()).take(1000).collect::<Vec<_>>();

        bench.iter(|| {
            test::black_box(p.iter().map(|&p| beta.inv_cdf(p)).collect::<Vec<_>>())
        });
    }
}
