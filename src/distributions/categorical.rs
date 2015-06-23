use Distribution;
use random::Generator;

/// A categorical distribution.
#[derive(Clone)]
pub struct Categorical {
    /// The size of the probabilty vector.
    pub k: usize,
    /// The probability vector.
    pub p: Vec<f64>,
}

impl Categorical {
    /// Create a categorical distribution with success probability `p`.
    ///
    /// It should hold that `p[i] >= 0`, `p[i] <= 1`, and `sum(p) == 1`.
    #[inline]
    pub fn new(p: &[f64]) -> Categorical {
        should!(is_probability_vector(p), {
            let mut in_unit = true;
            let mut sum = 0.0;
            for &p in p.iter() {
                if p < 0.0 || p > 1.0 {
                    in_unit = false;
                    break;
                }
                sum += p;
            }
            in_unit && (sum - 1.0).abs() <= 1e-12
        });
        Categorical { k: p.len(), p: p.to_vec() }
    }
}

impl Distribution for Categorical {
    type Value = usize;

    #[inline]
    fn mean(&self) -> f64 {
        self.p.iter().enumerate().fold(0.0, |sum, (i, p)| sum + i as f64 * p)
    }

    #[inline]
    fn var(&self) -> f64 {
        let mu = self.mean();
        self.p.iter().enumerate().fold(0.0, |sum, (i, p)| sum + (i as f64 - mu).powi(2) * p)
    }

    #[inline]
    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let sigma2 = self.var();
        let skew = self.p.iter().enumerate()
                         .fold(0.0, |sum, (i, p)| sum + (i as f64 - mu).powi(3) * p);
        skew / (sigma2 * sigma2.sqrt())
    }

    fn kurtosis(&self) -> f64 {
        let mu = self.mean();
        let sigma2 = self.var();
        let kurt = self.p.iter().enumerate()
                         .fold(0.0, |sum, (i, p)| sum + (i as f64 - mu).powi(4) * p);
        kurt / sigma2.powi(2) - 3.0
    }

    fn median(&self) -> f64 {
        if self.p[0] > 0.5 {
            return 0.0;
        } else if self.p[0] == 0.5 {
            return 0.5;
        }
        let mut sum = 0.0;
        for i in 0..self.k {
            sum += self.p[i];
            if sum == 0.5 {
                return (2 * i - 1) as f64 / 2.0;
            } else if sum > 0.5 {
                return i as f64;
            }
        }
        unreachable!()
    }

    fn modes(&self) -> Vec<usize> {
        let mut m = Vec::new();
        let mut max = 0.0;
        for (i, &p) in self.p.iter().enumerate() {
            if p == max {
                m.push(i);
            }
            if p > max {
                max = p;
                m = vec![i];
            }
        }
        m
    }

    #[inline]
    fn entropy(&self) -> f64 {
        -self.p.iter().fold(0.0, |sum, p| sum + p * p.ln())
    }

    #[inline]
    fn cdf(&self, x: usize) -> f64 {
        if x >= self.k - 1 { 1.0 }
        else {
            self.p.iter().take(x + 1).fold(0.0, |a, b| a + b)
        }
    }

    fn inv_cdf(&self, p: f64) -> usize {
        should!(0.0 <= p && p <= 1.0);
        if p == 0.0 {
            // return the first non-zero index
            return self.p.iter().enumerate().find(|&(_, &p)| p > 0.0).unwrap().0;
        }
        let mut sum = 0.0;
        for i in 0..self.k {
            sum += self.p[i];
            if sum >= p || sum == 1.0 {
                return i;
            }
        }
        self.k - 1
    }

    #[inline]
    fn pdf(&self, x: usize) -> f64 {
        self.p[x]
    }

    #[inline(always)]
    fn sample<G: Generator>(&self, generator: &mut G) -> usize {
        self.inv_cdf(generator.next::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use {Distribution, Sampler};
    use distributions::Categorical;

    macro_rules! new(
        (equal $k:expr) => { Categorical::new(&[1.0 / $k as f64; $k]) };
        ($p:expr) => { Categorical::new(&$p); }
    );

    #[test]
    fn mean() {
        assert_eq!(new!(equal 3).mean(), 1.0);
        assert_eq!(new!([0.3, 0.3, 0.4]).mean(), 1.1);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).mean(), 1.5);
    }

    #[test]
    fn var() {
        assert_eq!(new!(equal 3).var(), 2.0 / 3.0);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).var(), 11.0 / 12.0);
    }

    #[test]
    fn sd() {
        assert_eq!(new!(equal 2).sd(), 0.5);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).sd(), 0.9574271077563381);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(equal 6).skewness(), 0.0);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).skewness(), 0.0);
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
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).median(), 0.5);
    }

    #[test]
    fn modes() {
        assert_eq!(new!([0.6, 0.2, 0.2]).modes(), vec![0]);
        assert_eq!(new!(equal 2).modes(), vec![0, 1]);
        assert_eq!(new!(equal 3).modes(), vec![0, 1, 2]);
        assert_eq!(new!([0.4, 0.2, 0.4]).modes(), vec![0, 2]);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).modes(), vec![1, 2]);
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
        assert_eq!(&(0..4).map(|x| d1.pdf(x)).collect::<Vec<_>>(), &p.to_vec());

        let d2 = new!(equal 3);
        assert_eq!(&(0..3).map(|x| d2.pdf(x)).collect::<Vec<_>>(), &vec![1.0 / 3.0; 3])
    }

    #[test]
    fn cdf() {
        let d1 = new!([0.0, 0.75, 0.25, 0.0]);
        assert_eq!(&(0..4).map(|x| d1.cdf(x)).collect::<Vec<_>>(),
                   &vec![0.0, 0.75, 1.0, 1.0]);

        let d2 = new!(equal 3);
        assert_eq!(&(0..3).map(|x| d2.cdf(x)).collect::<Vec<_>>(),
                   &vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn inv_cdf() {
        let d1 = new!([0.0, 0.75, 0.25, 0.0]);
        let p1 = vec![0.0, 0.75, 0.7500001, 1.0];
        assert_eq!(&p1.iter().map(|&p| d1.inv_cdf(p)).collect::<Vec<_>>(), &vec![1, 1, 2, 2]);

        let d2 = new!(equal 3);
        let p2 = vec![0.0, 0.5, 0.75, 1.0];
        assert_eq!(&p2.iter().map(|&p| d2.inv_cdf(p)).collect::<Vec<_>>(), &vec![0, 1, 2, 2]);

    }

    #[test]
    fn sample() {
        let mut generator = ::generator();
        let sum = Sampler(&new!([0.0, 0.5, 0.5]), &mut generator).take(100).fold(0, |a, b| a + b);
        assert!(100 <= sum && sum <= 200);
    }
}
