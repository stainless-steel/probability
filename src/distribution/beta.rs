use distribution::Distribution;
use generator::Generator;

/// A beta distribution.
#[derive(Clone, Copy)]
pub struct Beta {
    alpha: f64,
    beta: f64,
    a: f64,
    b: f64,
    ln_beta: f64,
}

impl Beta {
    /// Create a beta distribution with shape parameters `alpha` and `beta`
    /// on interval `[a, b]`.
    ///
    /// It should hold that `alpha > 0`, `beta > 0`, and `a < b`.
    #[inline]
    pub fn new(alpha: f64, beta: f64, a: f64, b: f64) -> Beta {
        use special::ln_beta;
        should!(alpha > 0.0 && beta > 0.0 && a < b);
        Beta { alpha: alpha, beta: beta, a: a, b: b, ln_beta: ln_beta(alpha, beta) }
    }

    /// Return the first shape parameter.
    #[inline(always)] pub fn alpha(&self) -> f64 { self.alpha }

    /// Return the second shape parameter.
    #[inline(always)] pub fn beta(&self) -> f64 { self.beta }

    /// Return the left endpoint of the support.
    #[inline(always)] pub fn a(&self) -> f64 { self.a }

    /// Return the right endpoint of the support.
    #[inline(always)] pub fn b(&self) -> f64 { self.b }
}

impl Distribution for Beta {
    type Value = f64;

    #[inline]
    fn mean(&self) -> f64 {
        self.a + (self.b - self.a) * self.alpha / (self.alpha + self.beta)
    }

    #[inline]
    fn var(&self) -> f64 {
        let scale = self.b - self.a;
        let sum = self.alpha + self.beta;
        scale * scale * (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }

    #[inline]
    fn skewness(&self) -> f64 {
        let sum = self.alpha + self.beta;
        2.0 * (self.beta - self.alpha) * (sum + 1.0).sqrt() /
            ((sum + 2.0) * (self.alpha * self.beta).sqrt())
    }

    #[inline]
    fn kurtosis(&self) -> f64 {
        let sum = self.alpha + self.beta;
        let delta = self.alpha - self.beta;
        let product = self.alpha * self.beta;
        6.0 * (delta * delta * (sum + 1.0) - product * (sum + 2.0)) /
            (product * (sum + 2.0) * (sum + 3.0))
    }

    fn median(&self) -> f64 {
        match (self.alpha, self.beta) {
            (alpha, beta) if alpha == beta => 0.5 * (self.b - self.a),
            (alpha, beta) if alpha > 1.0 && beta > 1.0 => {
                self.a + (self.b - self.a) * (alpha - 1.0 / 3.0) / (alpha + beta - 2.0 / 3.0)
            },
            _ => self.inv_cdf(0.5),
        }
    }

    fn modes(&self) -> Vec<f64> {
        match (self.alpha, self.beta) {
            (1.0, 1.0) => vec![],
            (1.0, beta) if beta > 1.0 => vec![self.a],
            (alpha, 1.0) if alpha > 1.0 => vec![self.b],
            (alpha, beta) if alpha < 1.0 && beta < 1.0 => vec![self.a, self.b],
            (alpha, beta) if alpha < 1.0 && beta >= 1.0 => vec![self.a],
            (alpha, beta) if alpha >= 1.0 && beta < 1.0 => vec![self.b],
            (alpha, beta) => {
                vec![self.a + (self.b - self.a) * (alpha - 1.0) / (alpha + beta - 2.0)]
            },
        }
    }

    #[inline]
    fn entropy(&self) -> f64 {
        use special::digamma;
        let sum = self.alpha + self.beta;
        (self.b - self.a).ln() + self.ln_beta - (self.alpha - 1.0) * digamma(self.alpha) -
            (self.beta - 1.0) * digamma(self.beta) + (sum - 2.0) * digamma(sum)
    }

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        use special::inc_beta;
        inc_beta((x - self.a) / (self.b - self.a), self.alpha, self.beta, self.ln_beta)
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> f64 {
        use special::inv_inc_beta;
        should!(0.0 <= p && p <= 1.0);
        self.a + (self.b - self.a) * inv_inc_beta(p, self.alpha, self.beta, self.ln_beta)
    }

    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        let scale = self.b - self.a;
        let x = (x - self.a) / scale;
        ((self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() -
            self.ln_beta).exp() / scale
    }

    #[inline]
    fn sample<G>(&self, generator: &mut G) -> f64 where G: Generator {
        use distribution::gamma;
        let x = gamma::sample(self.alpha, 1.0, generator);
        let y = gamma::sample(self.beta, 1.0, generator);
        self.a + (self.b - self.a) * x / (x + y)
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($alpha:expr, $beta:expr, $a:expr, $b:expr) => (Beta::new($alpha, $beta, $a, $b));
    );

    #[test]
    fn mean() {
        assert_eq!(new!(0.5, 0.5, 0.0, 1.0).mean(), 0.5);
        assert_eq!(new!(0.0005, 0.9995, -1.0, 2.0).mean(), -0.9985);
    }

    #[test]
    fn var() {
        assert_eq!(new!(1.0, 1.0, 0.0, 1.0).var(), 1.0 / 12.0);
        assert_eq!(new!(2.0, 3.0, 0.0, 1.0).var(), 0.04);
        assert_eq!(new!(2.0, 3.0, -1.0, 2.0).var(), 0.36);
        assert_eq!(new!(5.0, 0.05, 0.0, 1.0).var(), new!(0.05, 5.0, 0.0, 1.0).var());
    }

    #[test]
    fn sd() {
        assert_eq!(new!(1.0, 1.0, 0.0, 1.0).sd(), (1f64 / 12.0).sqrt());
        assert_eq!(new!(2.0, 3.0, 0.0, 1.0).sd(), 0.2);
        assert_eq!(new!(2.0, 3.0, -1.0, 2.0).sd(), 0.6);
        assert_eq!(new!(5.0, 0.05, 0.0, 1.0).sd(), new!(0.05, 5.0, 0.0, 1.0).sd());
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(1.0, 1.0, 0.0, 1.0).skewness(), 0.0);
        assert_eq!(new!(2.0, 3.0, -1.0, 2.0).skewness(), 0.28571428571428575);
        assert_eq!(new!(3.0, 2.0, -1.0, 2.0).skewness(), -0.28571428571428575);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(1.0, 1.0, 0.0, 1.0).kurtosis(), -6.0 / 5.0);
        assert_eq!(new!(2.0, 3.0, -1.0, 2.0).kurtosis(), -0.6428571428571429);
        assert_eq!(new!(3.0, 2.0, -1.0, 2.0).kurtosis(), -0.6428571428571429);
    }

    #[test]
    fn median() {
        assert_eq!(new!(2.0, 2.0, 0.0, 1.0).median(), 0.5);
        assert_eq!(new!(2.0, 3.0, 0.0, 1.0).median(), 5.0 / 13.0);
        assert_eq!(new!(2.0, 3.0, -1.0, 2.0).median(), 3.0 * (5.0 / 13.0) -1.0);
    }

    #[test]
    fn modes() {
        let betas: [Beta; 9] = [
            new!(1.0, 1.0, -1.0, 2.0), new!(0.05, 0.05, -1.0, 2.0), new!(0.05, 5.0, -1.0, 2.0),
            new!(5.0, 0.05, -1.0, 2.0), new!(0.05, 3.0, -1.0, 2.0), new!(2.0, 0.05, -1.0, 2.0),
            new!(1.0, 3.0, -1.0, 2.0), new!(2.0, 1.0, -1.0, 2.0), new!(2.0, 3.0, -1.0, 2.0),
        ];
        let modes: [Vec<f64>; 9] = [
            vec![], vec![-1.0, 2.0], vec![-1.0],
            vec![2.0], vec![-1.0], vec![2.0],
            vec![-1.0], vec![2.0], vec![0.0],
        ];
        for (ref actual, expected) in betas.iter().map(|&beta| beta.modes()).zip(modes.iter()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn entropy() {
        use std::f64::consts::E;
        let betas = vec![
            new!(1.0, 1.0, 0.0, 1.0), new!(1.0, 1.0, 0.0, E),
            new!(2.0, 3.0, 0.0, 1.0), new!(2.0, 3.0, -1.0, 2.0),
        ];
        assert::close(&betas.iter().map(|beta| beta.entropy()).collect::<Vec<_>>(),
                      &vec![0.0, 1.0, -0.2349066497879999, 0.8637056388801096], 1e-15);
    }

    #[test]
    fn pdf() {
        let beta = new!(2.0, 3.0, -1.0, 2.0);
        let x = vec![
            -1.0, -0.85, -0.7, -0.55, -0.4, -0.25, -0.1, 0.05, 0.2, 0.35,
            0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0,
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
        assert::close(&x.iter().map(|&x| beta.pdf(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn cdf() {
        let beta = new!(2.0, 3.0, -1.0, 2.0);
        let x = vec![
            -1.0, -0.85, -0.7, -0.55, -0.4, -0.25, -0.1, 0.05, 0.2, 0.35,
            0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0,
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
        assert::close(&x.iter().map(|&x| beta.cdf(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn inv_cdf() {
        let beta = new!(1.0, 2.0, 3.0, 4.0);
        let p = vec![
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
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
        assert::close(&p.iter().map(|&p| beta.inv_cdf(p)).collect::<Vec<_>>(), &x, 1e-14);
    }

    #[test]
    fn sample() {
        for x in Sampler(&new!(1.0, 2.0, 7.0, 42.0), &mut generator()).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }
}
