#[allow(unused_imports)]
use special::Primitive;

use distribution;
use source::Source;

/// A continuous uniform distribution.
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    a: f64,
    b: f64,
}

impl Uniform {
    /// Create a uniform distribution on interval `[a, b]`.
    ///
    /// It should hold that `a < b`.
    #[inline]
    pub fn new(a: f64, b: f64) -> Self {
        should!(a < b);
        Uniform { a, b }
    }

    /// Return the left endpoint of the support.
    #[inline(always)]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Return the right endpoint of the support.
    #[inline(always)]
    pub fn b(&self) -> f64 {
        self.b
    }
}

impl Default for Uniform {
    #[inline]
    fn default() -> Self {
        Uniform::new(0.0, 1.0)
    }
}

impl distribution::Continuous for Uniform {
    #[inline]
    fn density(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (self.b - self.a)
        }
    }
}

impl distribution::Distribution for Uniform {
    type Value = f64;

    #[inline]
    fn distribution(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }
}

impl distribution::Entropy for Uniform {
    #[inline]
    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }
}

impl distribution::Inverse for Uniform {
    #[inline]
    fn inverse(&self, p: f64) -> f64 {
        should!((0.0..=1.0).contains(&p));
        self.a + (self.b - self.a) * p
    }
}

impl distribution::Kurtosis for Uniform {
    #[inline]
    fn kurtosis(&self) -> f64 {
        -1.2
    }
}

impl distribution::Mean for Uniform {
    #[inline]
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
}

impl distribution::Median for Uniform {
    #[inline]
    fn median(&self) -> f64 {
        use distribution::Mean;
        self.mean()
    }
}

impl distribution::Sample for Uniform {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        self.a + (self.b - self.a) * source.read::<f64>()
    }
}

impl distribution::Skewness for Uniform {
    #[inline]
    fn skewness(&self) -> f64 {
        0.0
    }
}

impl distribution::Variance for Uniform {
    #[inline]
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use prelude::*;

    macro_rules! new(
        ($a:expr, $b:expr) => (Uniform::new($a, $b));
    );

    #[test]
    fn distribution() {
        let d = new!(-1.0, 1.0);
        let x = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let p = vec![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0];

        assert_eq!(
            &x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(),
            &p
        );
    }

    #[test]
    fn density() {
        let d = new!(-1.0, 1.0);
        let x = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let p = vec![0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0];

        assert_eq!(&x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(), &p);
    }

    #[test]
    fn entropy() {
        use core::f64::consts::E;
        assert_eq!(new!(0.0, E).entropy(), 1.0);
    }

    #[test]
    fn inverse() {
        let d = new!(-1.0, 1.0);
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let p = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        assert_eq!(&p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(), &x);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(0.0, 2.0).kurtosis(), -1.2);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(0.0, 2.0).mean(), 1.0);
    }

    #[test]
    fn median() {
        assert_eq!(new!(0.0, 2.0).median(), 1.0);
    }

    #[test]
    fn sample() {
        for x in Independent(&new!(7.0, 42.0), &mut source::default([42, 69])).take(100) {
            assert!(7.0 <= x && x <= 42.0);
        }
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(0.0, 2.0).skewness(), 0.0);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(0.0, 12.0).variance(), 12.0);
    }
}
