use {Distribution, Generator};

/// A discrete Categorical distribution.
#[derive(Clone)]
pub struct Categorical {
    /// The size of the probabilty vector
    pub k: usize,
    /// The probability vector.
    pub p: Vec<f64>,
}

fn is_prob_vec(p: &[f64]) -> bool {
    let mut sum = 0.;
    for &p_i in p.iter() {
        if p_i < 0.|| p_i > 1. { return false; }
        sum += p_i;
    }
    (sum - 1.).abs() <= 1e-12
}

impl Categorical {
    /// Create a Categorical distribution with success probability `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p < 0` or `p > 1`.
    #[inline]
    pub fn new(p: &[f64]) -> Categorical {
        debug_assert!(is_prob_vec(p), "Categorical::new() is called with p not a probabilty vector");
        Categorical { k: p.len() as usize,  p: p.to_vec()}
    }
}

impl Distribution for Categorical {
    type Value = usize;

    #[inline]
    fn mean(&self) -> f64 {
        // sum_{i=0}^k i p_i
        self.p.iter().enumerate().fold(0., |acc, b| acc + b.0 as f64 * b.1)
    }

    #[inline]
    fn var(&self) -> f64 {
        let mu = self.mean();
        self.p.iter().enumerate().fold(
            0., |acc, b| acc + (b.0 as f64 - mu).powi(2) * b.1
        )
    }

    #[inline]
    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let sigma2 = self.var();
        let skew = self.p.iter().enumerate().fold(
            0., |acc, b| acc + (b.0 as f64 - mu).powi(3) * b.1
        );
        skew / (sigma2 * sigma2.sqrt())
    }

    #[inline]
    fn kurtosis(&self) -> f64 {
        let mu = self.mean();
        let sigma2 = self.var();
        let kurt = self.p.iter().enumerate().fold(
            0., |acc, b| acc + (b.0 as f64 - mu).powi(4) * b.1
        );
        kurt / sigma2.powi(2) - 3.
    }

    #[inline]
    fn median(&self) -> f64 {
        if self.p[0] > 0.5 { return 0.0; }
        else if self.p[0] == 0.5 { return 0.5; }
        let mut sum = 0.;
        for i in 0..self.k {
            sum += self.p[i];
            if sum == 0.5 { return (2 * i - 1) as f64 / 2.; }
            else if sum > 0.5 {
                return i as f64;
            }
        }
        unreachable!()
    }

    #[inline]
    fn modes(&self) -> Vec<Self::Value> {
        let mut m = Vec::new();
        let mut max = 0.;
        for (i, &p_i) in self.p.iter().enumerate() {
            if p_i == max { m.push(i); }
            if p_i > max {
                max = p_i;
                m = vec![i];
            }
        }
        m
    }

    #[inline]
    fn entropy(&self) -> f64 {
        - self.p.iter().fold(0., |acc, p_i| acc + p_i * p_i.ln())
    }

    #[inline]
    fn cdf(&self, x: Self::Value) -> f64 {
        if x >= self.k - 1 { 1. }
        else {
            self.p.iter().take(x + 1).fold(0., |a, b| a + b)
        }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> Self::Value {
        debug_assert!(0.0 <= p && p <= 1.0, "inv_cdf is called with p outside of [0, 1]");
        if p == 0. {
            // return the first non-zero index
            return self.p.iter().enumerate().find(|&v| *v.1 > 0.).unwrap().0;
        }
        let mut sum = 0.;
        for i in 0..self.k {
            sum += self.p[i];
            if sum >= p || sum == 1. { return i; }
        }
        self.k - 1
    }

    #[inline]
    fn pdf(&self, x: Self::Value) -> f64 {
        self.p[x]
    }

    #[inline]
    fn sample<G: Generator>(&self, generator: &mut G) -> Self::Value {
        self.inv_cdf(generator.gen::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use assert;

    use {Distribution, Sampler};
    use distributions::Categorical;

    macro_rules! new(
        (equal $k:expr) => { Categorical::new(&[1./$k as f64; $k]) };
        ($p:expr) => { Categorical::new(&$p); }
    );

    #[test]
    #[should_panic]
    fn invalid_prob_vec_1() {
        new!([0.6, 0.6, 0.6]);
    }

    #[test]
    #[should_panic]
    fn invalid_prob_vec_2() {
        new!([0.6, 0.6, 0.6, -0.2]);
    }

    #[test]
    #[should_panic]
    fn invalid_prob_vec_3() {
        new!([1.2, -0.2]);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(equal 3).mean(), 1.);
        assert_eq!(new!([0.3, 0.3, 0.4]).mean(), 1.1);
        assert_eq!(new!([1./6., 1./3., 1./3., 1./6.]).mean(), 1.5);
    }

    #[test]
    fn var() {
        assert_eq!(new!(equal 3).var(), 2./3.);
        assert_eq!(new!([1./6., 1./3., 1./3., 1./6.]).var(), 11./12.);
    }

    #[test]
    fn sd() {
        assert_eq!(new!(equal 2).sd(), 0.5);
        assert_eq!(new!([1./6., 1./3., 1./3., 1./6.]).sd(), 0.9574271077563381);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(equal 6).skewness(), 0.0);
        assert_eq!(new!([1./6., 1./3., 1./3., 1./6.]).skewness(), 0.);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).skewness(), -0.6);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(equal 2).kurtosis(), -2.0);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).kurtosis(), -0.7999999999999998);
    }

    #[test]
    fn median() {
        assert_eq!(new!([0.6, 0.2, 0.2]).median(), 0.0);
        assert_eq!(new!(equal 2).median(), 0.5);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).median(), 2.0);
        assert_eq!(new!([1./6., 1./3., 1./3., 1./6.]).median(), 0.5);
    }

    #[test]
    fn modes() {
        assert_eq!(new!([0.6, 0.2, 0.2]).modes(), vec![0]);
        assert_eq!(new!(equal 2).modes(), vec![0, 1]);
        assert_eq!(new!(equal 3).modes(), vec![0, 1, 2]);
        assert_eq!(new!([0.4, 0.2, 0.4]).modes(), vec![0, 2]);
        assert_eq!(new!([1./6., 1./3., 1./3., 1./6.]).modes(), vec![1, 2]);
    }

    #[test]
    fn entropy() {
        use std::f64::consts::LN_2;
        assert_eq!(new!(equal 2).entropy(), LN_2);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).entropy(), 1.2798542258336676);
    }

    #[test]
    fn pdf() {
        let p = [0.0, 0.75, 0.25, 0.0];
        let d1 = new!(p);
        assert::equal(&(0..4).map(|x| d1.pdf(x)).collect::<Vec<_>>(), &p.to_vec());

        let d2 = new!(equal 3);
        assert::equal(&(0..3).map(|x| d2.pdf(x)).collect::<Vec<_>>(), &vec![1./3.; 3])
    }

    #[test]
    fn cdf() {
        let d1 = new!([0.0, 0.75, 0.25, 0.0]);
        assert::equal(&(0..4).map(|x| d1.cdf(x)).collect::<Vec<_>>(),
                      &vec![0.0, 0.75, 1.0, 1.0]);

        let d2 = new!(equal 3);
        assert::equal(&(0..3).map(|x| d2.cdf(x)).collect::<Vec<_>>(),
                      &vec![1./3., 2./3., 1.]);
    }

    #[test]
    fn inv_cdf() {
        let d1 = new!([0.0, 0.75, 0.25, 0.0]);
        let p1 = vec![0.0, 0.75, 0.7500001, 1.0];
        assert::equal(&p1.iter().map(|&p| d1.inv_cdf(p)).collect::<Vec<_>>(),
                      &vec![1, 1, 2, 2]);

        let d2 = new!(equal 3);
        let p2 = vec![0.0, 0.5, 0.75, 1.0];
        assert::equal(&p2.iter().map(|&p| d2.inv_cdf(p)).collect::<Vec<_>>(),
                      &vec![0, 1, 2, 2]);

    }

    #[test]
    fn sample() {
        let mut generator = ::generator();

        // Discrete Uniform(1, 2)
        let sum = Sampler(&new!([0.0, 0.5, 0.5]), &mut generator)
            .take(100)
            .fold(0, |a, b| a + b);

        assert!(100 <= sum && sum <= 200);
    }
}
