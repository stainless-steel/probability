//! Probability distributions.

use source::Source;

/// A distribution.
pub trait Distribution {
    /// The type of outcomes.
    type Value;

    /// Compute the cumulative distribution function.
    fn cumulate(&self, f64) -> f64;

    /// Compute the cumulative distribution function.
    ///
    /// The function is an alias for `cumulate`.
    #[inline(always)]
    fn cdf(&self, x: f64) -> f64 {
        self.cumulate(x)
    }
}

/// A continuous distribution.
pub trait Continuous: Distribution {
    /// Compute the probability density function.
    fn density(&self, f64) -> f64;

    /// Compute the probability density function.
    ///
    /// The function is an alias for `density`.
    #[inline(always)]
    fn pdf(&self, x: f64) -> f64 {
        self.density(x)
    }
}

/// A discrete distribution.
pub trait Discrete: Distribution {
    /// Compute the probability mass function.
    fn mass(&self, Self::Value) -> f64;

    /// Compute the probability mass function.
    ///
    /// The function is an alias for `mass`.
    #[inline(always)]
    fn pmf(&self, x: Self::Value) -> f64 {
        self.mass(x)
    }
}

/// A distribution capable of computing the differential entropy.
pub trait Entropy: Distribution {
    /// Compute the differential entropy.
    ///
    /// The entropy is computed in nats.
    fn entropy(&self) -> f64;
}

/// A distribution capable of inverting the distribution function.
pub trait Inverse: Distribution {
    /// Compute the inverse of the distribution function.
    fn inv_cdf(&self, f64) -> Self::Value;
}

/// A distribution capable of computing the excess kurtosis.
pub trait Kurtosis: Skewness {
    /// Compute the excess kurtosis.
    fn kurtosis(&self) -> f64;
}

/// A distribution capable of computing the expected value.
///
/// The trait is applicable when the expected value exists, that is, finite.
pub trait Mean: Distribution {
    /// Compute the expected value.
    fn mean(&self) -> f64;
}

/// A distribution capable of computing the median.
///
/// The trait is applicable when exactly one median exists.
pub trait Median: Distribution {
    /// Compute the median.
    fn median(&self) -> f64;
}

/// A distribution capable of computing the modes.
///
/// The trait is applicable when the number of modes is finite.
pub trait Modes: Distribution {
    /// Compute the modes.
    fn modes(&self) -> Vec<Self::Value>;
}

/// A distribution capable of drawing samples.
pub trait Sample: Distribution {
    /// Draw a sample.
    fn sample<S>(&self, &mut S) -> Self::Value where S: Source;
}

/// A distribution capable of computing the skewness.
pub trait Skewness: Variance {
    /// Compute the skewness.
    fn skewness(&self) -> f64;
}

/// A distribution capable of computing the variance.
///
/// The trait is applicable when the variance exists, that is, finite.
pub trait Variance: Mean {
    /// Compute the variance.
    fn variance(&self) -> f64;

    /// Compute the standard deviation.
    #[inline(always)]
    fn deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute the standard deviation.
    ///
    /// The function is an alias for `deviation`.
    #[inline(always)]
    fn sd(&self) -> f64 {
        self.deviation()
    }

    /// Compute the variance.
    ///
    /// The function is an alias for `variance`.
    #[inline(always)]
    fn var(&self) -> f64 {
        self.variance()
    }
}

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::binomial::Binomial;
pub use self::categorical::Categorical;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::gaussian::Gaussian;
pub use self::uniform::Uniform;

mod bernoulli;
mod beta;
mod binomial;
mod categorical;
mod exponential;
mod gamma;
mod gaussian;
mod uniform;
