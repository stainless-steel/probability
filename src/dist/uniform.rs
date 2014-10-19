use std::rand::Rng;

use super::Distribution;

/// A continuous uniform distribution.
pub struct Uniform {
    /// The left endpoint of the support.
    pub a: f64,
    /// The right endpoint of the support.
    pub b: f64,
}

impl Uniform {
    /// Creates a uniform distribution on the interval `[a, b]`.
    #[inline]
    pub fn new(a: f64, b: f64) -> Uniform {
        Uniform { a: a, b: b }
    }
}

impl Distribution<f64> for Uniform {
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if self.b <= x {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    #[inline]
    fn inv_cdf(&self, p: f64) -> f64 {
        self.a + (self.b - self.a) * p
    }

    #[inline]
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        self.a + (self.b - self.a) * rng.gen()
    }
}

#[cfg(test)]
mod test {
    use super::super::{Distribution, Sampler};
    use super::Uniform;

    #[test]
    fn cdf() {
        let dist = Uniform::new(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(x.iter().map(|&x| dist.cdf(x)).collect::<Vec<_>>(), p);
    }

    #[test]
    fn inv_cdf() {
        let dist = Uniform::new(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(p.iter().map(|&p| dist.inv_cdf(p)).collect::<Vec<_>>(), x);
    }

    #[test]
    fn sample() {
        let mut rng = ::std::rand::task_rng();
        for x in Sampler(&Uniform::new(7.0, 42.0), &mut rng).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }
}
