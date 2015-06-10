//! Probability distributions.

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::categorical::Categorical;
pub use self::exponential::Exponential;
pub use self::gaussian::Gaussian;
pub use self::uniform::Uniform;

mod bernoulli;
mod beta;
mod categorical;
mod exponential;
mod gaussian;
mod uniform;
