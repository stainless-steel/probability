use distribution;
use random;

/// A categorical distribution.
#[derive(Clone)]
pub struct Categorical {
    k: usize,
    p: Vec<f64>,
    cumsum: Vec<f64>,
}

impl Categorical {
    /// Create a categorical distribution with success probability `p`.
    ///
    /// It should hold that `p[i] >= 0`, `p[i] <= 1`, and `sum(p) == 1`.
    pub fn new(p: &[f64]) -> Categorical {
        should!(is_probability_vector(p), {
            const EPSILON: f64 = 1e-12;
            p.iter().all(|&p| p >= 0.0 && p <= 1.0) &&
                (p.iter().fold(0.0, |sum, &p| sum + p) - 1.0).abs() < EPSILON
        });

        let k = p.len();
        let mut cumsum = p.to_vec();
        for i in 1..(k - 1) {
            cumsum[i] += cumsum[i - 1];
        }
        cumsum[k - 1] = 1.0;
        Categorical { k: k, p: p.to_vec(), cumsum: cumsum }
    }

    /// Return the number of categories.
    #[inline(always)]
    pub fn k(&self) -> usize { self.k }

    /// Return the event probabilities.
    #[inline(always)]
    pub fn p(&self) -> &[f64] { &self.p }
}

impl distribution::Distribution for Categorical {
    type Value = usize;

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let x = x as usize;
        if x >= self.k {
            return 1.0;
        }
        self.cumsum[x]
    }
}

impl distribution::Discrete for Categorical {
    #[inline]
    fn pmf(&self, x: usize) -> f64 {
        should!(x < self.k);
        self.p[x]
    }
}

impl distribution::Entropy for Categorical {
    fn entropy(&self) -> f64 {
        -self.p.iter().fold(0.0, |sum, p| sum + p * p.ln())
    }
}

impl distribution::Inverse for Categorical {
    fn inv_cdf(&self, p: f64) -> usize {
        should!(0.0 <= p && p <= 1.0);
        self.cumsum.iter().position(|&sum| sum > 0.0 && sum >= p).unwrap_or_else(|| {
            self.p.iter().rposition(|&p| p > 0.0).unwrap()
        })
    }
}

impl distribution::Kurtosis for Categorical {
    fn kurtosis(&self) -> f64 {
        use distribution::{Mean, Variance};
        let (mean, variance) = (self.mean(), self.variance());
        let kurt = self.p.iter().enumerate().fold(0.0, |sum, (i, p)| {
            sum + (i as f64 - mean).powi(4) * p
        });
        kurt / variance.powi(2) - 3.0
    }
}

impl distribution::Mean for Categorical {
    fn mean(&self) -> f64 {
        self.p.iter().enumerate().fold(0.0, |sum, (i, p)| sum + i as f64 * p)
    }
}

impl distribution::Median for Categorical {
    fn median(&self) -> f64 {
        if self.p[0] > 0.5 {
            return 0.0;
        }
        if self.p[0] == 0.5 {
            return 0.5;
        }
        for (i, &sum) in self.cumsum.iter().enumerate() {
            if sum == 0.5 {
                return (2 * i - 1) as f64 / 2.0;
            } else if sum > 0.5 {
                return i as f64;
            }
        }
        unreachable!()
    }
}

impl distribution::Modes for Categorical {
    fn modes(&self) -> Vec<usize> {
        let mut modes = Vec::new();
        let mut max = 0.0;
        for (i, &p) in self.p.iter().enumerate() {
            if p == max {
                modes.push(i);
            }
            if p > max {
                max = p;
                modes = vec![i];
            }
        }
        modes
    }
}

impl distribution::Sample for Categorical {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> usize where S: random::Source {
        use distribution::Inverse;
        self.inv_cdf(source.read::<f64>())
    }
}

impl distribution::Skewness for Categorical {
    fn skewness(&self) -> f64 {
        use distribution::{Mean, Variance};
        let (mean, variance) = (self.mean(), self.variance());
        let skew = self.p.iter().enumerate().fold(0.0, |sum, (i, p)| {
            sum + (i as f64 - mean).powi(3) * p
        });
        skew / (variance * variance.sqrt())
    }
}

impl distribution::Variance for Categorical {
    fn variance(&self) -> f64 {
        use distribution::Mean;
        let mean = self.mean();
        self.p.iter().enumerate().fold(0.0, |sum, (i, p)| {
            sum + (i as f64 - mean).powi(2) * p
        })
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    macro_rules! new(
        (equal $k:expr) => { Categorical::new(&[1.0 / $k as f64; $k]) };
        ($p:expr) => { Categorical::new(&$p); }
    );

    #[test]
    fn cdf() {
        let d = new!([0.0, 0.75, 0.25, 0.0]);
        let p = vec![0.0, 0.0, 0.75, 1.0, 1.0];

        let x = (-1..4).map(|x| d.cdf(x as f64)).collect::<Vec<_>>();
        assert_eq!(&x, &p);

        let x = (-1..4).map(|x| d.cdf(x as f64 + 0.5)).collect::<Vec<_>>();
        assert_eq!(&x, &p);

        let d = new!(equal 3);
        let p = vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];

        let x = (-1..3).map(|x| d.cdf(x as f64)).collect::<Vec<_>>();
        assert_eq!(&x, &p);

        let x = (-1..3).map(|x| d.cdf(x as f64 + 0.5)).collect::<Vec<_>>();
        assert_eq!(&x, &p);
    }

    #[test]
    fn pmf() {
        let p = [0.0, 0.75, 0.25, 0.0];
        let d = new!(p);
        assert_eq!(&(0..4).map(|x| d.pmf(x)).collect::<Vec<_>>(), &p.to_vec());

        let d = new!(equal 3);
        assert_eq!(&(0..3).map(|x| d.pmf(x)).collect::<Vec<_>>(), &vec![1.0 / 3.0; 3])
    }

    #[test]
    fn entropy() {
        use std::f64::consts::LN_2;
        assert_eq!(new!(equal 2).entropy(), LN_2);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).entropy(), 1.2798542258336676);
    }

    #[test]
    fn inv_cdf() {
        let d = new!([0.0, 0.75, 0.25, 0.0]);
        let p = vec![0.0, 0.75, 0.7500001, 1.0];
        assert_eq!(&p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>(), &vec![1, 1, 2, 2]);

        let d = new!(equal 3);
        let p = vec![0.0, 0.5, 0.75, 1.0];
        assert_eq!(&p.iter().map(|&p| d.inv_cdf(p)).collect::<Vec<_>>(), &vec![0, 1, 2, 2]);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(equal 2).kurtosis(), -2.0);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).kurtosis(), -0.7999999999999998);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(equal 3).mean(), 1.0);
        assert_eq!(new!([0.3, 0.3, 0.4]).mean(), 1.1);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).mean(), 1.5);
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
    fn sample() {
        let mut source = random::default();

        let sum = Independent(&new!([0.0, 0.5, 0.5]), &mut source).take(100).fold(0, |a, b| a + b);
        assert!(100 <= sum && sum <= 200);

        let p = (0..11).map(|i| if i % 2 != 0 { 0.2 } else { 0.0 }).collect::<Vec<_>>();
        assert!(Independent(&new!(p), &mut source).take(1000).all(|x| x % 2 != 0));
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(equal 6).skewness(), 0.0);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).skewness(), 0.0);
        assert_eq!(new!([0.1, 0.2, 0.3, 0.4]).skewness(), -0.6);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(equal 3).variance(), 2.0 / 3.0);
        assert_eq!(new!([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]).variance(), 11.0 / 12.0);
    }
}
