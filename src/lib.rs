//! A probability-theory toolbox.
//!
//! ## Example
//!
//! ```
//! use probability::prelude::*;
//!
//! let mut source = source::default();
//! let uniform = Uniform::new(0.0, 1.0);
//! let samples = Independent(&uniform, &mut source).take(10).collect::<Vec<_>>();
//! ```

#[cfg(test)]
extern crate assert;

extern crate random;
extern crate special;

macro_rules! should(
    ($requirement:expr) => (debug_assert!($requirement));
    ($requirement:expr, $code:expr) => (debug_assert!($code, stringify!($requirement)));
);

pub mod distribution;
pub mod prelude;
pub mod sampler;
pub mod source;
