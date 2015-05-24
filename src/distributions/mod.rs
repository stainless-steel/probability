//! Probability distributions.

pub use self::beta::Beta;
pub use self::exponential::Exponential;
pub use self::gaussian::Gaussian;
pub use self::uniform::Uniform;

mod beta;
mod exponential;
mod gaussian;
mod uniform;
