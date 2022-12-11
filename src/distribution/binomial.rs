use alloc::{vec, vec::Vec};
#[allow(unused_imports)]
use special::Primitive;

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
    /// For small `n`, a simple summation is utilized. For large `n` and large
    /// variances, a normal asymptotic approximation is used. Otherwise,
    /// Newton’s method is employed.
    ///
    /// ## References
    ///
    /// 1. S. Moorhead, “Efficient evaluation of the inverse binomial cumulative
    ///    distribution function where the number of trials is large,” Oxford
    ///    University, 2013.
    fn inverse(&self, p: f64) -> usize {
        use distribution::{Discrete, Distribution, Modes};

        should!((0.0..=1.0).contains(&p));

        macro_rules! sum_bottom_up(
            ($prod_term: expr) => ({
                let mut k = 1;
                let mut a = self.q.powi(self.n as i32);
                let mut sum = a - p;
                while sum < 0.0 {
                    a *= $prod_term(k);
                    sum += a;
                    k += 1;
                }
                k - 1
            });
        );
        macro_rules! sum_top_down(
            ($prod_term: expr) => ({
                let mut k = 1;
                let mut a = self.p.powi(self.n as i32);
                let mut sum = (1.0 - p) - a;
                while sum >= 0.0 {
                    a *= $prod_term(k);
                    sum -= a;
                    k += 1;
                }
                self.n - k + 1
            });
        );

        if p == 0.0 {
            0
        } else if p == 1.0 {
            self.n
        } else if self.n < 1000 {
            // Find if top-down or bottom-up summation is better.
            if p <= self.distribution((self.n / 2) as f64) {
                sum_bottom_up!(|k| self.p / self.q * ((self.n - k + 1) as f64 / k as f64))
            } else {
                sum_top_down!(|k| self.q / self.p * ((self.n - k + 1) as f64 / k as f64))
            }
        } else if self.npq > 80.0 {
            // Use a normal approximation.
            inverse_normal(self.p, self.np, self.npq, p).floor() as usize
        } else {
            // Use Newton’s method starting at the mode.
            const ALPHA: f64 = 0.999;
            let mut q = self.modes()[0] as f64;
            let mut alpha = 1.0;
            loop {
                let delta = alpha * (p - self.distribution(q)) / self.mass(q as usize);
                if delta.abs() < 0.5 {
                    return q as usize;
                }
                q += delta;
                alpha *= ALPHA;
            }
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

        if (self.np - self.np.trunc()) == 0.0 || (self.p == 0.5 && self.n % 2 != 0) {
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
        } else if (r - r.trunc()) != 0.0 {
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

// See [Moorhead, 2013, pp. 7].
#[rustfmt::skip]
fn inverse_normal(p: f64, np: f64, v: f64, u: f64) -> f64 {
    use distribution::gaussian;

    let w = gaussian::inverse(u);
    let w2 = w * w;
    let w3 = w2 * w;
    let w4 = w3 * w;
    let w5 = w4 * w;
    let w6 = w5 * w;
    let sd = v.sqrt();
    let sd_em1 = sd.recip();
    let sd_em2 = v.recip();
    let sd_em3 = sd_em1 * sd_em2;
    let sd_em4 = sd_em2 * sd_em2;
    let p2 = p * p;
    let p3 = p2 * p;
    let p4 = p2 * p2;

    np +
    sd * w +
    (p + 1.0) / 3.0 -
    (2.0 * p - 1.0) * w2 / 6.0 +
    sd_em1 * w3 * (2.0 * p2 - 2.0 * p - 1.0) / 72.0 -
    w * (7.0 * p2 - 7.0 * p + 1.0) / 36.0 +
    sd_em2 * (2.0 * p - 1.0) * (p + 1.0) * (p - 2.0) * (3.0 * w4 + 7.0 * w2 - 16.0 / 1620.0) +
    sd_em3 * (
        w5 * (4.0 * p4 - 8.0 * p3 - 48.0 * p2 + 52.0 * p - 23.0) / 17280.0 +
        w3 * (256.0 * p4 - 512.0 * p3 - 147.0 * p2 + 403.0 * p - 137.0) / 38880.0 -
        w * (433.0 * p4 - 866.0 * p3 - 921.0 * p2 + 1354.0 * p - 671.0) / 38880.0
    ) +
    sd_em4 * (
        w6 * (2.0 * p - 1.0) * (p2 - p + 1.0) * (p2 - p + 19.0) / 34020.0 +
        w4 * (2.0 * p - 1.0) * (9.0 * p4 - 18.0 * p3 - 35.0 * p2 + 44.0 * p - 25.0) / 15120.0 +
        w2 * (2.0 * p - 1.0) * (
                923.0 * p4 - 1846.0 * p3 + 5271.0 * p2 - 4348.0 * p + 5189.0
        ) / 408240.0 -
        4.0 * (2.0 * p - 1.0) * (p + 1.0) * (p - 2.0) * (23.0 * p2 - 23.0 * p + 2.0) / 25515.0
    )
    // + O(v.powf(-2.5)), with probabilty of 1 - 2e-9
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
        // Check edge cases.
        let d = new!(10, 0.5);
        assert_eq!(d.inverse(0.0), 0);
        assert_eq!(d.inverse(1.0), 10);

        // Check the summation.
        let d = new!(250, 0.55);
        assert_eq!(d.inverse(0.025), 122);
        assert_eq!(d.inverse(0.1), 127);

        // Check the normal approximation.
        let d = new!(2500, 0.55);
        assert_eq!(d.inverse(d.distribution(1298.0)), 1298);
        assert_eq!(new!(1001, 0.25).inverse(0.5), 250);
        assert_eq!(new!(1500, 0.15).inverse(0.2), 213);

        // Check Newton’s method.
        assert_eq!(new!(1_000_000, 2.5e-5).inverse(0.9995), 42);
        assert_eq!(new!(1_000_000_000, 6.66e-9).inverse(0.8), 8);
    }

    #[test]
    fn inverse_convergence() {
        let d = new!(1024, 0.009765625);
        assert_eq!(d.inverse(0.32185663510619567), 8);

        let d = new!(3666, 0.9810204628647335);
        assert_eq!(d.inverse(0.0033333333333332993), 3573);
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
