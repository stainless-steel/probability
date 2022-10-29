use alloc::{vec, vec::Vec};

use distribution;
use source::Source;

/// A binomial distribution.
#[derive(Clone, Copy, Debug)]
pub struct Binomial {
    n: usize,
    p: f64,
    q: f64,
    np: f64,
    nq: f64,
    npq: f64,
}

impl Binomial {
    /// Create a binomial distribution with `n` trails and success probability
    /// `p`.
    ///
    /// It should hold that `p >= 0` and `p <= 1`.
    pub fn new(n: usize, p: f64) -> Self {
        should!(0.0 < p && p < 1.0);
        let q = 1.0 - p;
        let np = n as f64 * p;
        let nq = n as f64 * q;
        Binomial {
            n,
            p,
            q,
            np,
            nq,
            npq: np * q,
        }
    }

    /// Create a binomial distribution with `n` trails and failure probability
    /// `q`.
    ///
    /// It should hold that if `q >= 0` or `q <= 1`. This constructor is
    /// preferable when `q` is very small.
    pub fn with_failure(n: usize, q: f64) -> Self {
        should!(0.0 < q && q < 1.0);
        let p = 1.0 - q;
        let np = n as f64 * p;
        let nq = n as f64 * q;
        Binomial {
            n,
            p,
            q,
            np,
            nq,
            npq: np * q,
        }
    }

    /// Return the number of trials.
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Return the success probability.
    #[inline(always)]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Return the failure probability.
    #[inline(always)]
    pub fn q(&self) -> f64 {
        self.q
    }
}

impl distribution::Discrete for Binomial {
    /// Compute the probability mass function.
    ///
    /// For large `n`, a saddle-point expansion is used for more accurate
    /// computation.
    ///
    /// ## References
    ///
    /// 1. C. Loader, “Fast and Accurate Computation of Binomial Probabilities,”
    ///    2000.
    fn mass(&self, x: usize) -> f64 {
        use core::f64::consts::PI;

        if self.p == 0.0 {
            return if x == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if x == self.n { 1.0 } else { 0.0 };
        }

        let n = self.n as f64;
        if x == 0 {
            (n * self.q.ln()).exp()
        } else if x == self.n {
            (n * self.p.ln()).exp()
        } else {
            let x = x as f64;
            let n_m_x = n - x;
            let ln_c = stirlerr(n)
                - stirlerr(x)
                - stirlerr(n_m_x)
                - ln_d0(x, self.np)
                - ln_d0(n_m_x, self.nq);
            ln_c.exp() * (n / (2.0 * PI * x * (n_m_x))).sqrt()
        }
    }
}

impl distribution::Distribution for Binomial {
    type Value = usize;

    /// Compute the cumulative distribution function.
    ///
    /// The implementation is based on the incomplete beta function.
    fn distribution(&self, x: f64) -> f64 {
        use special::Beta;
        if x < 0.0 {
            return 0.0;
        }
        let x = x as usize;
        if x == 0 {
            return self.q.powi(self.n as i32);
        }
        if x >= self.n {
            return 1.0;
        }
        let (p, q) = ((self.n - x) as f64, (x + 1) as f64);
        self.q.inc_beta(p, q, p.ln_beta(q))
    }
}

impl distribution::Entropy for Binomial {
    fn entropy(&self) -> f64 {
        use core::f64::consts::PI;
        use distribution::Discrete;

        if self.n > 10000 && self.npq > 80.0 {
            // Use a normal approximation.
            0.5 * ((2.0 * PI * self.npq).ln() + 1.0)
        } else {
            -(0..(self.n + 1)).fold(0.0, |sum, i| sum + self.mass(i) * self.mass(i).ln())
        }
    }
}

impl distribution::Inverse for Binomial {
    /// Compute the inverse of the cumulative distribution function.
    ///
    /// The code is based on a [C implementation][1] by John Burkardt.
    ///
    /// [1]: https://people.sc.fsu.edu/~jburkardt/c_src/prob/prob.html
    fn inverse(&self, p: f64) -> usize {
        use distribution::Discrete;

        should!((0.0..=1.0).contains(&p));
        if p == 0.0 {
            0
        } else if p == 1.0 {
            self.n
        } else {
            let mut x = 0;
            let mut q = 0.0;
            for y in 0..=self.n {
                q += self.mass(y);
                if p <= q {
                    x = y;
                    break;
                }
            }
            x
        }
    }
}

impl distribution::Kurtosis for Binomial {
    #[inline]
    fn kurtosis(&self) -> f64 {
        (1.0 - 6.0 * self.p * self.q) / self.npq
    }
}

impl distribution::Mean for Binomial {
    #[inline]
    fn mean(&self) -> f64 {
        self.np
    }
}

impl distribution::Median for Binomial {
    fn median(&self) -> f64 {
        use core::f64::consts::LN_2;
        use distribution::Inverse;

        if self.np.fract() == 0.0 || (self.p == 0.5 && self.n % 2 != 0) {
            self.np
        } else if self.p <= 1.0 - LN_2
            || self.p >= LN_2
            || (self.np.round() - self.np).abs() <= self.p.min(self.q)
        {
            self.np.round()
        } else if self.n > 1000 && self.npq > 80.0 {
            // Use a normal approximation.
            self.np.floor()
        } else {
            self.inverse(0.5) as f64
        }
    }
}

impl distribution::Modes for Binomial {
    fn modes(&self) -> Vec<usize> {
        let r = self.p * (self.n + 1) as f64;
        if r == 0.0 {
            vec![0]
        } else if self.p == 1.0 {
            vec![self.n]
        } else if r.fract() != 0.0 {
            vec![r.floor() as usize]
        } else {
            vec![r as usize - 1, r as usize]
        }
    }
}

impl distribution::Sample for Binomial {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> usize
    where
        S: Source,
    {
        use distribution::Inverse;
        self.inverse(source.read::<f64>())
    }
}

impl distribution::Skewness for Binomial {
    #[inline]
    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p) / self.npq.sqrt()
    }
}

impl distribution::Variance for Binomial {
    #[inline]
    fn variance(&self) -> f64 {
        self.npq
    }
}

// strilerr(n) = ln(n!) - ln(sqrt(2π * n) * (n / e)^n)
fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 1.0 / 12.0;
    const S1: f64 = 1.0 / 360.0;
    const S2: f64 = 1.0 / 1260.0;
    const S3: f64 = 1.0 / 1680.0;
    const S4: f64 = 1.0 / 1188.0;

    // See [Loader, 2000, pp. 7].
    #[allow(clippy::excessive_precision)]
    const SFE: [f64; 16] = [
        0.000000000000000000e+00,
        8.106146679532725822e-02,
        4.134069595540929409e-02,
        2.767792568499833915e-02,
        2.079067210376509311e-02,
        1.664469118982119216e-02,
        1.387612882307074800e-02,
        1.189670994589177010e-02,
        1.041126526197209650e-02,
        9.255462182712732918e-03,
        8.330563433362871256e-03,
        7.757367548795184079e-03,
        6.942840107209529866e-03,
        6.408994188004207068e-03,
        5.951370112758847736e-03,
        5.554733551962801371e-03,
    ];

    if n < 16.0 {
        return SFE[n as usize];
    }

    // See [Loader, 2000, eq. 4].
    let nn = n * n;
    if n > 500.0 {
        (S0 - S1 / nn) / n
    } else if n > 80.0 {
        (S0 - (S1 - S2 / nn) / nn) / n
    } else if n > 35.0 {
        (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    } else {
        (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
    }
}

// ln(np * D₀) = x * ln(x / np) + np - x
fn ln_d0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        // ε = (n / np) is close to 1. Use a series expansion.
        let mut s = (x - np).powi(2) / (x + np);
        let v = (x - np) / (x + np);
        let mut ej = 2.0 * x * v;
        let mut j = 1;
        loop {
            ej *= v * v;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s {
                return s1;
            }
            s = s1;
            j += 1;
        }
    }
    x * (x / np).ln() + np - x
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use assert;
    use prelude::*;

    macro_rules! new {
        ($n:expr, $p:expr) => {
            Binomial::new($n, $p)
        };
    }

    #[test]
    fn distribution() {
        let d = new!(16, 0.75);
        let p = vec![
            0.000000000000000e+00,
            2.328306436538699e-10,
            2.628657966852194e-07,
            3.810715861618527e-05,
            1.644465373829007e-03,
            2.712995628826319e-02,
            1.896545726340262e-01,
            5.950128899421541e-01,
            9.365235602017492e-01,
            1.000000000000000e+00,
        ];

        let x = (-1..9)
            .map(|i| d.distribution(2.0 * i as f64))
            .collect::<Vec<_>>();
        assert::close(&x, &p, 1e-14);

        let x = (-1..9)
            .map(|i| d.distribution(2.0 * i as f64 + 0.5))
            .collect::<Vec<_>>();
        assert::close(&x, &p, 1e-14);
    }

    #[test]
    fn entropy() {
        assert_eq!(new!(16, 0.25).entropy(), 1.9588018945068573);
        assert_eq!(new!(10_000_000, 0.5).entropy(), 8.784839178123887);
    }

    #[test]
    fn inverse() {
        let d = Binomial::new(250, 0.55);
        assert_eq!(d.inverse(0.0), 0);
        assert_eq!(d.inverse(0.025), 122);
        assert_eq!(d.inverse(0.1), 127);
        assert_eq!(d.inverse(1.0), 250);

        let x = 1298;
        let d = new!(2500, 0.55);
        assert_eq!(d.inverse(d.distribution(x as f64)), x);

        assert_eq!(new!(1001, 0.25).inverse(0.5), 250);
        assert_eq!(new!(1500, 0.15).inverse(0.2), 213);

        assert_eq!(new!(1_000_000, 2.5e-5).inverse(0.9995), 43);
        assert_eq!(new!(1_000_000_000, 6.66e-9).inverse(0.8), 9);
    }

    #[test]
    fn inverse_convergence() {
        let d = Binomial::new(3666, 0.9810204628647335);
        d.inverse(0.0033333333333332993);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(16, 0.25).kurtosis(), -0.041666666666666664);
    }

    #[test]
    fn mass() {
        let d = new!(16, 0.25);
        let p = vec![
            1.002259575761855e-02,
            1.336346101015806e-01,
            2.251990651711821e-01,
            1.100973207503558e-01,
            1.966023584827779e-02,
            1.359226182103156e-03,
            3.432389348745344e-05,
            2.514570951461788e-07,
            2.328306436538698e-10,
        ];

        assert::close(
            &(0..9).map(|i| d.mass(2 * i)).collect::<Vec<_>>(),
            &p,
            1e-14,
        );
    }

    #[test]
    fn mean() {
        assert_eq!(new!(16, 0.25).mean(), 4.0);
    }

    #[test]
    fn median() {
        assert_eq!(new!(16, 0.25).median(), 4.0);
        assert_eq!(new!(3, 0.5).median(), 1.5);
        assert_eq!(new!(1000, 0.015).median(), 15.0);
        assert_eq!(new!(39, 0.1).median(), 4.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(16, 0.25).modes(), vec![4]);
        assert_eq!(new!(3, 0.5).modes(), vec![1, 2]);
        assert_eq!(new!(1000, 0.015).modes(), vec![15]);
        assert_eq!(new!(39, 0.1).modes(), vec![3, 4]);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(16, 0.25).skewness(), 0.2886751345948129);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(16, 0.25).variance(), 3.0);
    }
}
