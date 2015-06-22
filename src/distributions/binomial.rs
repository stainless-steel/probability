use {Distribution, Generator};

/// A binomial distribution.
#[derive(Clone, Copy)]
pub struct Binomial {
    /// The number of trials.
    pub n: i32,
    /// The success probability.
    pub p: f64,
    /// The probability of failure.
    pub q: f64,
    np: f64,
    nq: f64,
    npq: f64,
}

impl Binomial {
    /// Create a binomial distribution with `n` observations and
    /// success probability `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p < 0` or `p > 1` or if n < 0
    #[inline]
    pub fn new(n: i32, p: f64) -> Binomial {
        debug_assert!(n >= 0, "Binomial::new() is called with n < 0");
        debug_assert!(0. < p && p < 1., "Binomial::new() is called with p < 0 or p > 1");
        let q = 1.0 - p;
        let np = n as f64 * p;
        let nq = n as f64 * q;
        Binomial {
            n: n,
            p: p,
            q: q,
            np: np,
            nq: nq,
            npq: np * q,
        }
    }

    /// Create a binomial distribution with `n` observations and
    /// failure probability `q`.
    ///
    /// Use this one instead of `Binomial::new()` if `q` is very
    /// small.
    ///
    /// # Panics
    ///
    /// Panics if `q < 0` or `q > 1` or if n < 0
    #[inline]
    pub fn new_failprob(n: i32, q: f64) -> Binomial {
        debug_assert!(n >= 0, "Binomial::new() is called with n < 0");
        debug_assert!(0. < q && q < 1., "Binomial::new() is called with q < 0 or p > 1");
        let p = 1.0 - q;
        let np = n as f64 * p;
        let nq = n as f64 * q;
        Binomial {
            n: n,
            p: p,
            q: q,
            np: np,
            nq: nq,
            npq: np * q,
        }
    }
}

impl Distribution for Binomial {
    type Value = i32;

    #[inline]
    fn mean(&self) -> f64 { self.np }

    #[inline]
    fn var(&self) -> f64 { self.npq }

    #[inline]
    fn skewness(&self) -> f64 {
        (1. - 2. * self.p) / (self.npq).sqrt()
    }

    #[inline]
    fn kurtosis(&self) -> f64 {
        (1. - 6. * self.p * self.q) / (self.npq)
    }

    #[inline]
    fn median(&self) -> f64 {
        use std::f64::consts::LN_2;
        if self.np.fract() == 0. {
            self.np
        } else if self.p == 0.5 && self.n % 2 != 0 {
            self.np
        } else if self.p <= 1. - LN_2 || self.p >= LN_2 ||
                  (self.np.round() - self.np).abs() <= self.p.min(self.q) {
            self.np.round()
        } else if self.n > 1000 && self.npq > 80. {
            // Normal approximation.
            self.np.floor()
        } else {
            self.inv_cdf(0.5) as f64
        }
    }

    #[inline]
    fn modes(&self) -> Vec<Self::Value> {
        let r = self.p * (self.n + 1) as f64;

        if r == 0. { vec![0] }
        else if self.p == 1. { vec![self.n] }
        else if r.fract() != 0. { vec![r.floor() as Self::Value] }
        else {
            let r_int = r as Self::Value;
            vec![r_int - 1, r_int]
        }
    }

    #[inline]
    fn entropy(&self) -> f64 {
        use std::f64::consts::PI;

        if self.n > 10000 && self.npq > 80. {
            // Use normal approximation.
            0.5 * ((2. * PI * self.npq).ln() + 1.)
        } else {
            // Calculate directly.
            -(0..self.n+1).fold(0., |sum, i| sum + self.pdf(i) * self.pdf(i).ln())
        }
    }

    /// Compute the cumulative distribution function (CDF) at point `x`.
    ///
    /// Uses the incomplete beta function.
    #[inline]
    fn cdf(&self, x: Self::Value) -> f64 {
        use special::{inc_beta, ln_beta};
        if x == 0 {
            return self.pdf(0);
        } else if x >= self.n {
            return 1.
        }
        let n_m_x = (self.n - x) as f64;
        let x_p_1 = (x + 1) as f64;
        inc_beta(self.q, n_m_x, x_p_1, ln_beta(n_m_x, x_p_1))
    }

    /// Compute the inverse of the cumulative distribution function at
    /// probability `p`.
    ///
    /// For small `n` we use simple summation. For larger `n` and
    /// large variance we use normal asymptotic approximation. Else we
    /// use the Newton method to search.
    ///
    /// # References
    ///
    /// 1. Sean Moorhead (2013). “Efficient evaluation of the inverse
    ///    Binomial cumulative distribution function where the number
    ///    of trials is large”.
    fn inv_cdf(&self, p: f64) -> Self::Value {
        debug_assert!(0.0 <= p && p <= 1.0, "inv_cdf is called with p outside of [0, 1]");

        // Rename p as to not be confused with self.p.
        let u = p;

        macro_rules! buttom_up_sum {
            ($prod_term: expr) => {
                {
                    let mut k = 1;
                    let mut a = self.q.powi(self.n);
                    let mut sum = a - u;
                    while sum < 0. {
                        a *= $prod_term(k);
                        sum += a;
                        k += 1;
                    }
                    k - 1
                }
            };
        }
        macro_rules! top_down_sum {
            ($prod_term: expr) => {
                {
                    let mut k = 1;
                    let mut a = self.p.powi(self.n);
                    let mut sum = (1. - u) - a;
                    while sum >= 0. {
                        a *= $prod_term(k);
                        sum -= a;
                        k += 1;
                    }
                    self.n - k + 1
                }
            };
        }

        // See Moorhead (2013) pp. 7.
        let normal_approx = |p: f64, np: f64, v:f64| -> f64 {
            use distributions::Gaussian;
            let w = Gaussian::new(0., 1.).inv_cdf(u);
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

            (np + sd * w + ((p + 1.) / 3. - (2.*p - 1.) * w2 / 6.)
             + sd_em1 * w3 * (2.*p2 - 2.*p - 1.) / 72. - w * (7.*p2 - 7.*p + 1.) / 36.
             + sd_em2 * (2.*p - 1.) * (p + 1.) * (p - 2.) * (3.*w4 + 7.*w2 - 16. / 1620.)
             + sd_em3 * (w5 * (4.*p4 - 8.*p3 - 48.*p2 + 52.*p - 23.) / 17280.
                         + w3 * (256.*p4 - 512.*p3 - 147.*p2 + 403.*p - 137.) / 38880.
                         - w * (433.*p4 - 866.*p3 - 921.*p2 + 1354.*p - 671.) / 38880.)
             + sd_em4 * (w6 * (2.*p - 1.) * (p2 - p + 1.) * (p2 - p + 19.) / 34020.
                         + w4 * (2.*p - 1.) * (9.*p4 - 18.*p3 - 35.*p2 + 44.*p - 25.) / 15120.
                         + w2 * ((2.*p - 1.) * (923.*p4 - 1846.*p3 + 5271.*p2 - 4348.*p + 5189.)
                                 / 408240.)
                         - 4. * ((2.*p - 1.) * (p + 1.) * (p - 2.) * (23.*p2 - 23.*p + 2.)
                                 / 25515.))) // + O(v.powf(-2.5)), with probabilty of 1 - 2e-9.
        };

        if u == 1. {
            self.n
        } else if u == 0. {
            0
        } else if self.n < 1000 {
            // Find if top-down or buttom-up sumation is better.
            if u <= self.cdf(self.n / 2) {
                buttom_up_sum!(|k| self.p / self.q * ((self.n - k + 1) as f64 / k as f64))
            } else {
                top_down_sum!(|k| self.q / self.p * ((self.n - k + 1) as f64 / k as f64))
            }
        } else if self.npq > 80. {
            // Use normal asymptotic approximation.
            let approx = normal_approx(self.p, self.np, self.npq);
            approx.floor() as Self::Value
        } else {
            // Use newton to search, starting at the mode
            let modes = self.modes();
            let mut m = modes[0];
            loop {
                let next = (u - self.cdf(m)) / self.pdf(m);
                if -0.5 < next && next < 0.5 { break; }
                m += next.round() as Self::Value;
            }
            m
        }
    }

    /// Compute the probability density function (PDF) at point `x`.
    ///
    /// Uses saddle-point expansion[1] for more accurate computation
    /// for large n.
    ///
    /// # References
    ///
    /// 1. Catherine Loader (2000). “Fast and Accurate Computation of
    ///    Binomial Probabilities”.
    fn pdf(&self, x: Self::Value) -> f64 {
        use std::f64::consts::PI;

        let n = self.n as f64;

        // strilerr(n) =  ln(n!) - ln(sqrt(2π * n) * (n/e)^n)
        fn stirlerr(n: f64) -> f64 {
            const S0: f64 = 1. / 12.;
            const S1: f64 = 1. / 360.;
            const S2: f64 = 1. / 1260.;
            const S3: f64 = 1. / 1680.;
            const S4: f64 = 1. / 1188.;

            // Precomputed values for the first n = 0, ⋯, 15
            // see Loader (2000) pp. 7.
            const SFE: [f64; 16] = [
                0.000000000000000000e+00, 8.106146679532725822e-02,
                4.134069595540929409e-02, 2.767792568499833915e-02,
                2.079067210376509311e-02, 1.664469118982119216e-02,
                1.387612882307074800e-02, 1.189670994589177010e-02,
                1.041126526197209650e-02, 9.255462182712732918e-03,
                8.330563433362871256e-03, 7.757367548795184079e-03,
                6.942840107209529866e-03, 6.408994188004207068e-03,
                5.951370112758847736e-03, 5.554733551962801371e-03,
                ];

            let nn = n * n;
            if n < 16. { SFE[n as usize] }
            // For all other n, use decreasing number of terms in the
            // Stirling-De Moivre series expansion, see in Loader (2000)
            // eq. 4.
            else if n > 500. { (S0 - S1 / nn) / n }
            else if n > 80. { (S0 - (S1 - S2 / nn) / nn) / n }
            else if n > 35. { (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n }
            else { (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n }
        }


        // Log of the deviance term: ln(np*D₀) = x ln(x/np) + np - x.
        fn ln_d0(x: f64, np: f64) -> f64 {
            if (x - np).abs() < 0.1 * (x + np) {
                // ε = (n / np) is close to 1. Use series expansion.
                let mut s = (x - np).powi(2) / (x + np);
                let v = (x - np) / (x + np);
                let mut ej = 2. * x * v;
                let mut j = 1;
                loop {
                    ej *= v * v;
                    let s1 = s + ej / (2 * j + 1) as f64;
                    if s1 == s { return s1; }
                    s = s1;
                    j += 1;
                }
            }
            x * (x / np).ln() + np - x
        }

        if self.p == 0. { if x == 0 { 1. } else { 0. } }
        else if self.p == 1. { if x == self.n { 1. } else { 0. } }
        else if x == 0 { (n * self.q.ln()).exp() }
        else if x == self.n { (n * self.p.ln()).exp() }
        else {
            let x = x as f64;
            let n_m_x = n - x;
            let ln_c = stirlerr(n) - stirlerr(x) - stirlerr(n_m_x)
                - ln_d0(x, self.np) - ln_d0(n_m_x, self.nq);
            ln_c.exp() * (n / (2. * PI * x * (n_m_x))).sqrt()
        }
    }

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> Self::Value {
        self.inv_cdf(generator.gen::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use assert;

    use Distribution;
    use distributions::Binomial;

    macro_rules! new {
        ($n:expr, $p:expr) => (Binomial::new($n, $p));
        (fail $n:expr, $p:expr) => (Binomial::new_failprob($n, $p));
    }

    #[test]
    #[should_panic]
    fn invalid_trails() {
        new!(-2, 0.5);
    }

    #[test]
    #[should_panic]
    fn invalid_success_probability_1() {
        new!(16, 2.0);
    }

    #[test]
    #[should_panic]
    fn invalid_success_probability_2() {
        new!(16, -0.5);
    }

    #[test]
    #[should_panic]
    fn invalid_failure_probability_1() {
        new!(fail 16, 2.0);
    }

    #[test]
    #[should_panic]
    fn invalid_failure_probability_2() {
        new!(fail 16, -0.5);
    }

    #[test]
    fn new() {
        new!(16, 0.5);
    }

    #[test]
    fn new_failure() {
        new!(fail 16, 1e-24);
    }

    #[test]
    fn mean() { assert_eq!(new!(16, 0.25).mean(), 4.); }

    #[test]
    fn var() { assert_eq!(new!(16, 0.25).var(), 3.); }

    #[test]
    fn sd() { assert_eq!(new!(16, 0.5).sd(), 2.); }

    #[test]
    fn skewness() { assert_eq!(new!(16, 0.25).skewness(), 0.2886751345948129); }

    #[test]
    fn kurtosis() { assert_eq!(new!(16, 0.25).kurtosis(), -0.041666666666666664); }

    #[test]
    fn median() {
        assert_eq!(new!(16, 0.25).median(), 4.);
        assert_eq!(new!(3, 0.5).median(), 1.5);
        assert_eq!(new!(1000, 0.015).median(), 15.);
        assert_eq!(new!(39, 0.1).median(), 4.);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(16, 0.25).modes(), vec![4]);
        assert_eq!(new!(3, 0.5).modes(), vec![1, 2]);
        assert_eq!(new!(1000, 0.015).modes(), vec![15]);
        assert_eq!(new!(39, 0.1).modes(), vec![3, 4]);
    }

    #[test]
    fn entropy() {
        assert_eq!(new!(16, 0.25).entropy(), 1.9588018945068573);
        assert_eq!(new!(10_000_000, 0.5).entropy(), 8.784839178123887);
    }

    #[test]
    fn pdf() {
        let binom = new!(16, 0.25);
        let probs = vec![
            1.002259575761855e-02, 1.336346101015806e-01, 2.251990651711821e-01,
            1.100973207503558e-01, 1.966023584827779e-02, 1.359226182103156e-03,
            3.432389348745344e-05, 2.514570951461788e-07, 2.328306436538698e-10,
        ];

        assert::close(&(0..9).map(|i| binom.pdf(2*i)).collect::<Vec<_>>(), &probs, 1e-14);
    }

    #[test]
    fn cdf() {
        let binom = new!(16, 0.75);
        let probs = vec![
            2.328306436538699e-10, 2.628657966852194e-07, 3.810715861618527e-05,
            1.644465373829007e-03, 2.712995628826319e-02, 1.896545726340262e-01,
            5.950128899421541e-01, 9.365235602017492e-01, 1.000000000000000e+00,
        ];
        assert::close(&(0..9).map(|i| binom.cdf(2*i)).collect::<Vec<_>>(), &probs, 1e-14);
    }

    #[test]
    fn inv_cdf() {
        let binom = Binomial::new(250, 0.55);
        assert_eq!(binom.inv_cdf(0.1), 127);
        assert_eq!(binom.inv_cdf(0.025), 122);

        let x = 1298;
        let binom2 = new!(2500, 0.55);
        assert_eq!(binom2.inv_cdf(binom2.cdf(x)), x);
        assert_eq!(new!(1001, 0.25).inv_cdf(0.5), 250);
        assert_eq!(new!(1500, 0.15).inv_cdf(0.2), 213);

        assert_eq!(new!(1_000_000, 2.5e-5).inv_cdf(0.9995), 42);
        assert_eq!(new!(1_000_000_000, 6.66e-9).inv_cdf(0.8), 8);
    }
}
