use distribution;
use source::Source;

/// A triangular distribution.
#[derive(Debug, Clone, Copy)]
pub struct Triangular {
    a: f64,
    b: f64,
    c: f64,
}

impl Triangular {
    /// Create a triangular distribution with mode `c` on interval `[a, b]`.
    ///
    /// It should hold that `a < b`, `a <= c`, and `c <= b`.
    #[inline]
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        should!(a < b && a <= c && c <= b);
        Triangular { a: a, b: b, c: c }
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

    /// Return the mode parameter.
    #[inline(always)]
    pub fn c(&self) -> f64 {
        self.c
    }
}

impl distribution::Continuous for Triangular {
    fn density(&self, x: f64) -> f64 {
        nonnan!(x);
        let &Triangular { a, b, c } = self;
        if x < a || b < x {
            0.0
        } else {
            let mut factor = 2.0 / (b - a);
            if x < c {
                factor *= (x - a) / (c - a);
            } else if x > c {
                factor *= (b - x) / (b - c);
            }
            factor
        }
    }
}

impl distribution::Distribution for Triangular {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        nonnan!(x);
        let &Triangular { a, b, c } = self;
        if x <= a {
            0.0
        } else if b <= x {
            1.0
        } else {
            let diff = b - a;
            if x <= c {
                (x - a).powi(2) / diff / (c - a)
            } else {
                1.0 - (b - x).powi(2) / diff / (b - c)
            }
        }
    }
}

impl distribution::Entropy for Triangular {
    #[inline]
    fn entropy(&self) -> f64 {
        0.5 + ((self.b - self.a) / 2.0).ln()
    }
}

impl distribution::Inverse for Triangular {
    fn inverse(&self, p: f64) -> f64 {
        should!(0.0 <= p && p <= 1.0);
        nonnan!(p);
        let &Triangular { a, b, c } = self;
        if p == 0.0 {
            a
        } else if p == 1.0 {
            b
        } else {
            let p0 = (c - a) / (b - a);
            if p < p0 {
                ((b - a) * (c - a) * p).sqrt() + a
            } else if p > p0 {
                b - ((b - a) * (b - c) * (1.0 - p)).sqrt()
            } else {
                c
            }
        }
    }
}

impl distribution::Kurtosis for Triangular {
    #[inline]
    fn kurtosis(&self) -> f64 {
        -(3.0 / 5.0)
    }
}

impl distribution::Mean for Triangular {
    #[inline]
    fn mean(&self) -> f64 {
        (self.a + self.b + self.c) / 3.0
    }
}

impl distribution::Median for Triangular {
    fn median(&self) -> f64 {
        let &Triangular { a, b, c } = self;
        if c >= (a + b) / 2.0 {
            a + ((b - a) * (c - a) / 2.0).sqrt()
        } else {
            b - ((b - a) * (b - c) / 2.0).sqrt()
        }
    }
}

impl distribution::Modes for Triangular {
    #[inline]
    fn modes(&self) -> Vec<f64> {
        vec![self.c]
    }
}

impl distribution::Sample for Triangular {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64
    where
        S: Source,
    {
        use distribution::Inverse;
        self.inverse(source.read::<f64>())
    }
}

impl distribution::Skewness for Triangular {
    fn skewness(&self) -> f64 {
        let &Triangular { a, b, c } = self;
        let npart = (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c);
        let dpart = a * a + b * b + c * c - a * b - a * c - b * c;
        (2f64.sqrt() * npart) / (5.0 * dpart.powf(3.0 / 2.0))
    }
}

impl distribution::Variance for Triangular {
    fn variance(&self) -> f64 {
        let &Triangular { a, b, c } = self;
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($a:expr, $b:expr, $c:expr) => (Triangular::new($a, $b, $c));
    );

    #[test]
    fn density() {
        let d = new!(1.0, 5.0, 3.0);
        let x = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];
        let p = vec![0.0, 0.0, 0.125, 0.25, 0.375, 0.5, 0.375, 0.25, 0.125, 0.0, 0.0];

        assert::close(
            &x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(),
            &p,
            1e-15,
        );
    }

    #[test]
    fn distribution() {
        let d = new!(1.0, 5.0, 3.0);
        let x = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];
        let p = vec![0.0, 0.0, 0.03125, 0.125, 0.28125, 0.5, 0.71875, 0.875, 0.96875, 1.0, 1.0];

        assert::close(
            &x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(),
            &p,
            1e-15,
        );
    }

    #[test]
    fn entropy() {
        let c = 0.5f64.exp();
        assert_eq!(new!(0.0, 2.0 * c, c).entropy(), 1.0);
    }

    #[test]
    fn inverse() {
        let d = new!(1.0, 5.0, 3.0);
        let p = vec![0.0, 0.03125, 0.125, 0.28125, 0.5, 0.71875, 0.875, 0.96875, 1.0];
        let x = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];

        assert::close(
            &p.iter().map(|&p| d.inverse(p)).collect::<Vec<_>>(),
            &x,
            1e-14,
        );
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(1.0, 5.0, 3.0).kurtosis(), -(3.0 / 5.0));
    }

    #[test]
    fn mean() {
        assert_eq!(new!(1.0, 5.0, 3.0).mean(), 3.0);
    }

    #[test]
    fn median() {
        assert_eq!(new!(1.0, 5.0, 3.0).median(), 3.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(1.0, 5.0, 3.0).modes(), vec![3.0]);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(1.0, 5.0, 3.0).skewness(), 0.0);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(1.0, 5.0, 3.0).variance(), (12.0 / 18.0));
    }

    #[test]
    fn deviation() {
        assert_eq!(new!(1.0, 5.0, 3.0).deviation(), (12f64 / 18.0).sqrt());
    }
}
