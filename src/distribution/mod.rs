//! Probability distributions.

use random::Source;

/// A probability distribution.
pub trait Distribution {
    /// The type of outcomes.
    type Value;

    /// Compute the distribution function.
    ///
    /// The function is also known as the cumulative distribution function.
    fn cdf(&self, f64) -> f64;
}

/// A continuous distribution.
pub trait Continuous: Distribution {
    /// Compute the probability density function.
    fn pdf(&self, f64) -> f64;
}

/// A discrete distribution.
pub trait Discrete: Distribution {
    /// Compute the probability mass function.
    fn pmf(&self, Self::Value) -> f64;
}

/// A distribution capable of computing the differential entropy.
pub trait Entropy: Distribution {
    /// Compute the differential entropy.
    ///
    /// The entropy is computed in nats.
    fn entropy(&self) -> f64;
}

/// A distribution capable of computing the expected value.
///
/// The trait is applicable when the expected value exists, that is, finite.
pub trait Expectation: Distribution {
    /// Compute the expected value.
    fn expectation(&self) -> f64;

    /// Compute the expected value.
    ///
    /// The function is an alias for `expectation`.
    #[inline(always)]
    fn mean(&self) -> f64 {
        self.expectation()
    }
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

/// A distribution capable of sampling.
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
pub trait Variance: Expectation {
    /// Compute the variance.
    fn variance(&self) -> f64;

    /// Compute the standard deviation.
    #[inline(always)]
    fn deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute the variance.
    ///
    /// The function is an alias for `variance`.
    #[inline(always)]
    fn var(&self) -> f64 {
        self.variance()
    }

    /// Compute the standard deviation.
    ///
    /// The function is an alias for `deviation`.
    #[inline(always)]
    fn sd(&self) -> f64 {
        self.deviation()
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
