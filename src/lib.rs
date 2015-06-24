//! A probability-theory toolbox.

#[cfg(test)] extern crate assert;
extern crate special;

macro_rules! should(
    ($requirement:expr) => ({
        debug_assert!($requirement, stringify!($requirement))
    });
    ($requirement:expr, $code:expr) => ({
        debug_assert!($code, stringify!($requirement))
    });
);

pub mod distribution;
pub mod generator;
pub mod sampler;

pub mod prelude;
