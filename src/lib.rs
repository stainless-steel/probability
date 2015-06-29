//! A probability-theory toolbox.
//!
//! ## Example
//!
//! ```
//! use probability::prelude::*;
//!
//! let uniform = Uniform::new(0.0, 1.0);
//! let mut generator = generator::default();
//! let samples = Independent(&uniform, &mut generator).take(10).collect::<Vec<_>>();
//! ```

#[cfg(test)]
extern crate assert;

extern crate special;

macro_rules! should(
    ($requirement:expr) => ({
        debug_assert!($requirement);
    });
    ($requirement:expr, $code:expr) => ({
        debug_assert!($code, stringify!($requirement))
    });
);

pub mod distribution;
pub mod generator;
pub mod prelude;
pub mod sampler;
